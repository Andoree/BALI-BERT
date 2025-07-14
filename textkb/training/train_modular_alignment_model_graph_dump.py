import itertools
import logging
import os.path
import re
import time
from typing import Dict, Tuple, List

import deepspeed
import pandas as pd
import torch.cuda
from pytorch_metric_learning import losses
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from textkb.data.offsets_dataset import TextGraphGraphNeighborsOffsetDataset
from textkb.modeling.bi_encoder_model import ModularAlignmentModelBiEncoder
from textkb.modeling.graph_encoders import GATv2Encoder
from textkb.modeling.model import AlignmentModel, ModularAlignmentModel
from textkb.training.training import train_model
from textkb.utils.io import create_dir_if_not_exists, save_dict, load_dict, load_tokenized_concepts, load_offset_index, \
    load_adjacency_lists
from textkb.utils.model_utils import save_model_checkpoint
from textkb.utils.utils import set_random_seed

# deepspeed.init_distributed()

CHECKPOINT_NAME_PATTERN_STR = r"checkpoint_e_(?P<epoch_id>\d+)_steps_(?P<step_id>\d+)\.pth"


def model_step(model, token_is_entity_mask, entity_node_ids, subtoken2entity_edge_index, concept_graph_input,
               concept_graph_edge_index, num_entities, corrupted_sentence_input, token_labels, amp) -> Dict:
    if amp:
        with autocast():
            ld = model(
                corrupted_sentence_input=corrupted_sentence_input,
                token_is_entity_mask=token_is_entity_mask,
                entity_node_ids=entity_node_ids,
                subtoken2entity_edge_index=subtoken2entity_edge_index,
                concept_graph_input=concept_graph_input,
                concept_graph_edge_index=concept_graph_edge_index,
                num_entities=num_entities,
                token_labels=token_labels)
    else:
        ld = model(
            corrupted_sentence_input=corrupted_sentence_input,
            token_is_entity_mask=token_is_entity_mask,
            entity_node_ids=entity_node_ids,
            subtoken2entity_edge_index=subtoken2entity_edge_index,
            concept_graph_input=concept_graph_input,
            concept_graph_edge_index=concept_graph_edge_index,
            num_entities=num_entities,
            token_labels=token_labels)

    return ld


def alignment_model_train_step(model: ModularAlignmentModel, batch, amp, device) -> Dict:
    corrupted_sentence_input = [t.to(device) for t in batch["corrupted_sentence_input"]]
    concept_graph_input = [t.to(device) for t in batch["node_input"]]
    token_labels = batch["token_labels"].to(device)
    token_is_entity_mask = batch["token_is_entity_mask"].to(device)
    entity_node_ids = batch["entity_node_ids"].to(device)
    subtoken2entity_edge_index = batch["subtoken2entity_edge_index"].to(device)
    concept_graph_edge_index = batch["concept_graph_edge_index"].to(device)
    num_entities = batch["num_entities"]

    losses_dict = model_step(model=model,
                             corrupted_sentence_input=corrupted_sentence_input,
                             token_is_entity_mask=token_is_entity_mask,
                             entity_node_ids=entity_node_ids,
                             subtoken2entity_edge_index=subtoken2entity_edge_index,
                             num_entities=num_entities,
                             token_labels=token_labels,
                             concept_graph_input=concept_graph_input,
                             concept_graph_edge_index=concept_graph_edge_index,
                             amp=amp)

    return losses_dict


def alignment_model_val_step(model: AlignmentModel, batch, amp, device) -> Dict:
    corrupted_sentence_input = [t.to(device) for t in batch["corrupted_sentence_input"]]
    concept_graph_input = [t.to(device) for t in batch["node_input"]]
    token_labels = batch["token_labels"].to(device)
    token_is_entity_mask = batch["token_is_entity_mask"].to(device)
    entity_node_ids = batch["entity_node_ids"].to(device)
    subtoken2entity_edge_index = batch["subtoken2entity_edge_index"].to(device)
    concept_graph_edge_index = batch["concept_graph_edge_index"].to(device)
    num_entities = batch["num_entities"]
    model.eval()
    with torch.no_grad():
        losses_dict = model_step(model=model,
                                 corrupted_sentence_input=corrupted_sentence_input,
                                 token_is_entity_mask=token_is_entity_mask,
                                 entity_node_ids=entity_node_ids,
                                 subtoken2entity_edge_index=subtoken2entity_edge_index,
                                 num_entities=num_entities,
                                 token_labels=token_labels,
                                 concept_graph_input=concept_graph_input,
                                 concept_graph_edge_index=concept_graph_edge_index,
                                 amp=amp)

    return losses_dict


def alignment_model_val_epoch(model, val_loader: DataLoader, amp, device, **kwargs):
    model.eval()
    losses_dict = {}
    global_num_samples = 0
    logging.info("Evaluating...")
    pbar = tqdm(val_loader, miniters=len(val_loader) // 100, total=len(val_loader), mininterval=10.0)
    for batch in pbar:
        num_samples = len(batch["corrupted_sentence_input"][0])
        step_ld = alignment_model_val_step(model=model, batch=batch, amp=amp, device=device)
        for loss_name, loss_tensor in step_ld.items():
            if losses_dict.get(loss_name) is None:
                losses_dict[loss_name] = 0.
            losses_dict[loss_name] += float(step_ld[loss_name]) * num_samples
            # total_loss = sum(float(v) * num_samples for v in losses_dict.values())
            # losses_dict["total_loss"] = total_loss
        global_num_samples += num_samples

    for k in losses_dict.keys():
        v = losses_dict[k]
        losses_dict[k] = v / global_num_samples
    total_loss = sum(losses_dict.values())
    losses_dict["total_loss"] = total_loss

    model.train()

    return losses_dict


def alignment_model_train_epoch(model, train_loader: DataLoader, val_loader: DataLoader, eval_every_n_steps: int,
                                save_chkpnt_step_interval: int, optimizer: torch.optim.Optimizer, scaler, scheduler,
                                amp, deepspeed_flag, device, step_loss_file_path, initial_global_num_steps: int,
                                local_rank: int, **kwargs):
    output_dir = os.path.dirname(step_loss_file_path)
    model.train()
    rank_flag = local_rank in (-1, 0)
    if deepspeed_flag:
        logging.info(f"Synchronizing before epoch start. Proc {local_rank} reached.")
        deepspeed.comm.barrier()
    step_losses_dict = {"total_loss": [], "learning_rate": []}
    epoch_losses_dict = {"total_loss": 0., }
    if rank_flag:
        pbar = tqdm(train_loader, miniters=len(train_loader) // 100, total=len(train_loader))
    else:
        pbar = train_loader

    for in_batch_step_id, batch in enumerate(pbar):
        if len(batch) == 0:
            continue
        if not deepspeed_flag:
            optimizer.zero_grad()

        step_ld = alignment_model_train_step(model=model, batch=batch, amp=amp, device=device)
        total_loss = torch.sum(torch.stack(tuple(step_ld.values())))

        losses_s = ', '.join(f"{k}: {float(v):.5f}" for k, v in step_ld.items())
        losses_s = (f"Total loss: {float(total_loss):.5f}. "
                    f"{losses_s}. LR: {float(optimizer.param_groups[0]['lr']):.10f}")
        step_ld["total_loss"] = total_loss
        if rank_flag and in_batch_step_id % 10 == 0:
            pbar.set_description(losses_s)
        if not deepspeed_flag:
            if amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            scheduler.step()
        else:
            model.backward(total_loss)
            model.step()
        step_losses_dict["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if eval_every_n_steps is not None and (
                initial_global_num_steps + in_batch_step_id) % eval_every_n_steps == 0 and (
                initial_global_num_steps + in_batch_step_id) > 0:

            if rank_flag:
                if not deepspeed_flag:
                    optimizer.zero_grad()
                logging.info(f"Evaluating at step: {initial_global_num_steps + in_batch_step_id}...")
                val_losses_dict = alignment_model_val_epoch(model, val_loader, amp, device, **kwargs)
                # total_loss = sum(val_losses_dict.values())
                # total_loss = torch.sum(torch.stack(tuple(val_losses_dict.values())))
                losses_s = ', '.join(f"{k}: {float(v):.5f}" for k, v in val_losses_dict.items())
                losses_s = f"Validation losses. {losses_s}."
                logging.info(f"Step {initial_global_num_steps + in_batch_step_id}, {losses_s}.")
            deepspeed.comm.barrier()
        if (save_chkpnt_step_interval is not None
                and (initial_global_num_steps + in_batch_step_id) % save_chkpnt_step_interval == 0):
            logging.info(f"Saving checkpoint at step: {initial_global_num_steps + in_batch_step_id}...")
            save_model_checkpoint(model, epoch_id=(initial_global_num_steps + in_batch_step_id) // len(train_loader),
                                  num_steps=initial_global_num_steps + in_batch_step_id, optimizer=optimizer,
                                  scheduler=scheduler, scaler=scaler, output_dir=output_dir,
                                  deepspeed_flag=deepspeed_flag, rank_flag=rank_flag)
        for k, v in step_ld.items():
            if epoch_losses_dict.get(k) is None:
                epoch_losses_dict[k] = 0.
            if step_losses_dict.get(k) is None:
                step_losses_dict[k] = []
            epoch_losses_dict[k] += float(v)
            step_losses_dict[k].append(float(v))
        # epoch_losses_dict["total_loss"] += float(total_loss)
        # step_losses_dict["total_loss"].append(float(total_loss))

    epoch_losses_df = pd.DataFrame(step_losses_dict)
    epoch_losses_df.to_csv(step_loss_file_path, sep='\t')

    epoch_losses_dict = {key: lo / (in_batch_step_id + 1e-9) for key, lo in epoch_losses_dict.items()}

    return epoch_losses_dict, in_batch_step_id


def main():
    # TODO: ОБНОВИТЬ SUB-ДИРЕКТОРИЮ!!
    # TODO: В основной скрит добавить проверку на то, что do_lower_case одинаковый для токенизации concept_names
    # TODO: и предложений!!!!

    # from textkb.utils.arg_parsing import parse_modular_biencoder_alignment_model_args
    # args = parse_modular_biencoder_alignment_model_args()
    from textkb.utils.arg_parsing_debug import parse_modular_biencoder_alignment_model_args_debug
    args = parse_modular_biencoder_alignment_model_args_debug()

    print(args)
    local_rank = args.local_rank
    # os.environ["MASTER_PORT"] = "8888"
    set_random_seed(args.random_seed)
    output_dir = args.output_dir
    if local_rank in (-1, 0):
        create_dir_if_not_exists(output_dir)
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )

    use_cuda = args.use_cuda
    deepspeed_flag = args.deepspeed
    if deepspeed_flag:
        deepspeed_cfg_path = args.deepspeed_cfg_path
    if use_cuda and not torch.cuda.is_available():
        raise Exception(f"Provided use_cuda flag but no CUDA GPU is available")
    if use_cuda and deepspeed_flag:
        device = torch.device("cuda")
    elif use_cuda and not deepspeed_flag:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logging.info(f"Using torch device: {device}")
    mlm_task = args.mlm_task
    contrastive_loss_name = args.contrastive_loss
    textual_contrastive_task = args.textual_contrastive_task
    text_node_contrastive_task = args.text_node_contrastive_task
    sentence_emb_pooling = args.sentence_emb_pooling
    concept_emb_pooling = args.concept_emb_pooling

    concept_name_masking_prob = args.concept_name_masking_prob
    mention_masking_prob = args.mention_masking_prob
    mlm_probability = args.mlm_probability

    gat_num_layers = args.gat_num_layers
    gat_num_hidden_channels = args.gat_num_hidden_channels
    gat_dropout_p = args.gat_dropout_p
    gat_num_att_heads = args.gat_num_att_heads
    gat_attention_dropout_p = args.gat_attention_dropout_p
    gat_add_self_loops = args.gat_add_self_loops
    use_rel = args.use_rel
    max_n_neighbors = args.max_n_neighbors

    graph_data_dir = args.graph_data_dir
    tokenized_concepts_path = args.tokenized_concepts_path
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir

    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")
    num_train_batches = sum(1 for e in os.listdir(train_data_dir))

    rel2id_path = os.path.join(graph_data_dir, "rel2id")
    rel2id = load_dict(rel2id_path, dtype_1=str, dtype_2=int)
    num_rels = len(rel2id)

    unique_concept_idx_path = os.path.join(graph_data_dir, "mentioned_concepts_idx")
    # mentioned_concept_ids = load_list_elem_per_line(input_path=unique_concept_idx_path, dtype=int)
    # global2local_concept_id = {global_id: local_id for local_id, global_id in enumerate(mentioned_concept_ids)}

    bert_encoder_name = args.bert_encoder_name
    sentence_max_length = args.sentence_max_length
    concept_max_length = args.concept_max_length
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    save_every_n_steps = args.save_every_N_steps
    eval_every_n_steps = args.eval_every_N_steps
    if not args.ignore_tokenization_assert:
        train_sent_cfg_path = os.path.join(train_data_dir, "config.txt")
        train_sent_token_cfg = load_dict(train_sent_cfg_path, sep='\t')
        assert int(train_sent_token_cfg["sentence_max_length"]) == sentence_max_length
        assert train_sent_token_cfg["transformer_tokenizer_name"] == bert_encoder_name

        if val_data_dir is not None:
            val_sent_cfg_path = os.path.join(val_data_dir, "config.txt")
            val_sent_token_cfg = load_dict(val_sent_cfg_path, sep='\t')
            assert int(val_sent_token_cfg["sentence_max_length"]) == sentence_max_length
            assert val_sent_token_cfg["transformer_tokenizer_name"] == bert_encoder_name

    amp_flag = args.amp
    if deepspeed_flag and amp_flag:
        raise RuntimeError("Using AMP with DeepSpeed is not supported. AMP is already implemented in DeepSpeed.")

    multigpu_flag = args.parallel
    save_every_N_epoch = args.save_every_N_epoch
    bert_lr = args.bert_learning_rate
    graph_lr = args.graph_learning_rate
    max_num_warmup_steps = args.max_num_warmup_steps
    warmup_steps_ratio = args.warmup_steps_ratio
    weight_decay = args.weight_decay

    contrastive_loss_weight = args.contrastive_loss_weight
    model_checkpoint_dir = args.model_checkpoint_path

    mlm_s = "_mlm" if mlm_task else ""

    cl_name_s = f"_{contrastive_loss_name}" if (textual_contrastive_task or text_node_contrastive_task) else ""
    ttcl_s = f"_ttcl" if textual_contrastive_task else ""
    tgcl_s = f"_tgcl" if text_node_contrastive_task else ""

    genc_s = (f"_gat-{gat_num_layers}-{gat_num_hidden_channels}-{gat_num_att_heads}-{gat_dropout_p}"
              f"-{gat_attention_dropout_p}-{'loops' if gat_add_self_loops else 'noloops'}")

    output_subdir = (f"contr_l_{contrastive_loss_weight}{mlm_s}{ttcl_s}{tgcl_s}{cl_name_s}{genc_s}"
                     f"_pool_s-{sentence_emb_pooling}_c-{concept_emb_pooling}_mask-{mention_masking_prob}"
                     f"-{concept_name_masking_prob}_blr_{bert_lr}_graph_lr_{graph_lr}_b_{batch_size}")
    output_dir = os.path.join(output_dir, output_subdir)
    if local_rank in (-1, 0):
        create_dir_if_not_exists(output_dir)

    node_id2input_ids: List[Tuple[Tuple[int]]] = load_tokenized_concepts(tok_conc_path=tokenized_concepts_path)
    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, use_rel)

    bert_encoder = AutoModel.from_pretrained(bert_encoder_name)
    sentence_bert_tokenizer = AutoTokenizer.from_pretrained(bert_encoder_name)
    bert_hidden_size = bert_encoder.config.hidden_size

    if contrastive_loss_name == "ms-loss":
        contrastive_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    elif contrastive_loss_name == "infonce":
        contrastive_loss = losses.NTXentLoss(temperature=0.07)
    else:
        raise NotImplementedError(f"unsupported contrastive loss: {contrastive_loss_name}")

    graph_encoder = GATv2Encoder(in_channels=bert_hidden_size,
                                 num_layers=gat_num_layers,
                                 num_hidden_channels=gat_num_hidden_channels,
                                 dropout_p=gat_dropout_p,
                                 num_att_heads=gat_num_att_heads,
                                 attention_dropout_p=gat_attention_dropout_p,
                                 add_self_loops=gat_add_self_loops,
                                 multigpu=False)

    alignment_model = ModularAlignmentModelBiEncoder(bert_encoder=bert_encoder,
                                                     graph_encoder=graph_encoder,
                                                     bert_tokenizer=sentence_bert_tokenizer,
                                                     multigpu=multigpu_flag,
                                                     mlm_task=mlm_task,
                                                     textual_contrastive_task=textual_contrastive_task,
                                                     text_node_contrastive_task=text_node_contrastive_task,
                                                     textual_contrastive_loss=contrastive_loss,
                                                     text_graph_contrastive_loss=contrastive_loss,
                                                     sentence_emb_pooling=sentence_emb_pooling,
                                                     concept_emb_pooling=concept_emb_pooling,
                                                     device=device)
    # TODO: Если в директории есть config, то каким будет num_train_batches?
    # TODO: Пойти переделать subsample 20k - надо перенумеровать файлы
    tr_offsets, tr_offset_lowerbounds, tr_offset_upperbounds, tr_offset_filenames = load_offset_index(train_data_dir)
    logging.info(f"Loaded train offset index. {len(tr_offsets)} samples, {len(tr_offset_lowerbounds)} files.")
    train_dataset = TextGraphGraphNeighborsOffsetDataset(tokenizer=sentence_bert_tokenizer,
                                                         input_data_dir=train_data_dir,
                                                         offsets=tr_offsets,
                                                         offset_lowerbounds=tr_offset_lowerbounds,
                                                         offset_upperbounds=tr_offset_upperbounds,
                                                         offset_filenames=tr_offset_filenames,
                                                         node_id2adjacency_list=node_id2adjacency_list,
                                                         node_id2input_ids=node_id2input_ids,
                                                         max_n_neighbors=max_n_neighbors,
                                                         use_rel=use_rel,
                                                         sentence_max_length=sentence_max_length,
                                                         concept_max_length=concept_max_length,
                                                         mlm_probability=mlm_probability,
                                                         concept_name_masking_prob=concept_name_masking_prob,
                                                         mention_masking_prob=mention_masking_prob)

    init_checkpoint_step_id = 0
    init_epoch_id = 0
    if model_checkpoint_dir is not None:
        checkpoint_name = model_checkpoint_dir.rstrip('/').split("/")[-1]

        m = re.match(CHECKPOINT_NAME_PATTERN_STR, checkpoint_name)
        assert m is not None
        init_epoch_id = int(m.group("epoch_id")) - 1
        init_checkpoint_step_id = int(m.group("step_id"))
        logging.info(f"Training will resume from epoch {init_epoch_id} (global step {init_checkpoint_step_id})")
    if init_checkpoint_step_id == 0:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.dataloader_num_workers,
                                      shuffle=True, collate_fn=train_dataset.collate_fn)
    else:
        assert init_checkpoint_step_id > 0
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.dataloader_num_workers,
                                      shuffle=True, collate_fn=train_dataset.collate_fn)

    val_loader = None
    val_epoch_fn = None
    if val_data_dir is not None:
        num_val_batches = sum(1 for e in os.listdir(val_data_dir))
        val_offsets, val_offset_lowerbounds, val_offset_upperbounds, val_offset_filenames = load_offset_index(
            val_data_dir)
        logging.info(f"Loaded val offset index. {len(val_offsets)} samples, {len(val_offset_lowerbounds)} files.")
        val_dataset = TextGraphGraphNeighborsOffsetDataset(tokenizer=sentence_bert_tokenizer,
                                                           input_data_dir=val_data_dir,
                                                           offsets=val_offsets,
                                                           offset_lowerbounds=val_offset_lowerbounds,
                                                           offset_upperbounds=val_offset_upperbounds,
                                                           offset_filenames=val_offset_filenames,
                                                           node_id2adjacency_list=node_id2adjacency_list,
                                                           node_id2input_ids=node_id2input_ids,
                                                           max_n_neighbors=max_n_neighbors,
                                                           use_rel=use_rel,
                                                           sentence_max_length=sentence_max_length,
                                                           concept_max_length=concept_max_length,
                                                           mlm_probability=mlm_probability,
                                                           mention_masking_prob=mention_masking_prob,
                                                           concept_name_masking_prob=concept_name_masking_prob)

        val_loader = DataLoader(val_dataset, batch_size=batch_size // 4, num_workers=args.dataloader_num_workers,
                                shuffle=False, collate_fn=val_dataset.collate_fn)
        val_epoch_fn = alignment_model_val_epoch

    logging.info(f"Using {bert_lr} LR for BERT and {graph_lr} for graph encoder")

    non_bert_modules = [module for name, module in alignment_model._modules.items() if name != 'bert_encoder']
    non_bert_params = itertools.chain(*(m.parameters() for m in non_bert_modules))
    optimizer = torch.optim.AdamW([{"params": alignment_model.bert_encoder.parameters()},
                                   {"params": non_bert_params, "lr": graph_lr}],
                                  lr=bert_lr, weight_decay=weight_decay)

    # opt_params = [{"params": alignment_model.graph_encoder.parameters(), "lr": graph_lr},
    #               {"params": alignment_model.bert_encoder.parameters(), "lr": bert_lr},
    #               ]
    # optimizer = torch.optim.AdamW(opt_params, weight_decay=weight_decay)
    num_batches = len(train_dataloader)
    overall_num_steps = num_batches * num_epochs - init_checkpoint_step_id

    num_warmup_steps = min(int(warmup_steps_ratio * overall_num_steps), max_num_warmup_steps)
    logging.info(f"LR scheduler parameters: {num_warmup_steps} warm-up steps, {overall_num_steps} total num steps")
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=overall_num_steps)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    if deepspeed_flag:
        alignment_model, optimizer, train_dataloader, scheduler = deepspeed.initialize(model=alignment_model,
                                                                                       training_data=train_dataset,
                                                                                       optimizer=optimizer,
                                                                                       collate_fn=train_dataset.collate_fn,
                                                                                       config=deepspeed_cfg_path,
                                                                                       lr_scheduler=scheduler)

    start = time.time()
    train_model(model=alignment_model, optimizer=optimizer, train_epoch_fn=alignment_model_train_epoch,
                val_epoch_fn=val_epoch_fn, chkpnt_path=model_checkpoint_dir, train_loader=train_dataloader,
                val_loader=val_loader, num_epochs=num_epochs, output_dir=output_dir,
                contrastive_loss_weight=contrastive_loss_weight, scaler=scaler, scheduler=scheduler,
                save_chkpnt_epoch_interval=save_every_N_epoch, save_chkpnt_step_interval=save_every_n_steps,
                init_checkpoint_step_id=init_checkpoint_step_id, eval_every_n_steps=eval_every_n_steps,
                init_checkpoint_epoch_id=init_epoch_id, amp=amp_flag, deepspeed_flag=deepspeed_flag,
                device=device, local_rank=local_rank)

    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    logging.info(f"Training Time took {training_hour} hours {training_minute} minutes {training_second} seconds")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main()
