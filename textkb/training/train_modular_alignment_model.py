import itertools
import logging
import os.path
import re
import time

import numpy as np
import pandas as pd
import torch.cuda
from pytorch_metric_learning import losses
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import deepspeed
# deepspeed.init_distributed()


from textkb.data.precomputed_graph_embs_dataset import CachedPrecomputedGraphTextDataset
from textkb.modeling.model import AlignmentModel, ModularAlignmentModel
from textkb.training.training import train_model
from textkb.utils.io import create_dir_if_not_exists, save_dict, load_dict, load_list_elem_per_line
from textkb.utils.model_utils import save_model_checkpoint
from textkb.utils.utils import set_random_seed

CHECKPOINT_NAME_PATTERN_STR = r"checkpoint_e_(?P<epoch_id>\d+)_steps_(?P<step_id>\d+)\.pth"


def alignment_model_train_step(model: ModularAlignmentModel, batch, amp, device):
    sentence_input = [t.to(device) for t in batch["sentence_input"]]
    corrupted_sentence_input = [t.to(device) for t in batch["corrupted_sentence_input"]]
    token_labels = batch["token_labels"].to(device)
    token_is_entity_mask = batch["token_is_entity_mask"].to(device)
    entity_node_ids = batch["entity_node_ids"].to(device)
    subtoken2entity_edge_index = batch["subtoken2entity_edge_index"].to(device)
    num_entities = batch["num_entities"]
    pos_triples = batch["pos_triples"].to(device)
    neg_node_ids = batch["neg_node_ids"].to(device)
    has_edge_mask = batch["has_edge_mask"].to(device)
    link_pred_batch_type = batch["batch_type"]

    losses_dict = model_step(model=model,
                             sentence_input=sentence_input,
                             token_is_entity_mask=token_is_entity_mask,
                             entity_node_ids=entity_node_ids,
                             subtoken2entity_edge_index=subtoken2entity_edge_index,
                             num_entities=num_entities,
                             corrupted_sentence_input=corrupted_sentence_input,
                             token_labels=token_labels,
                             pos_triples=pos_triples,
                             neg_node_ids=neg_node_ids,
                             has_edge_mask=has_edge_mask,
                             link_pred_batch_type=link_pred_batch_type,
                             amp=amp)

    return losses_dict


def model_step(model, sentence_input, token_is_entity_mask, entity_node_ids, subtoken2entity_edge_index,
               num_entities, corrupted_sentence_input, token_labels, pos_triples, neg_node_ids, has_edge_mask,
               link_pred_batch_type, amp):
    if amp:
        with autocast():
            masked_lm_loss, contrastive_loss, graph_lp_loss, intermodal_lp_loss = model.forward(
                sentence_input=sentence_input,
                token_is_entity_mask=token_is_entity_mask,
                entity_node_ids=entity_node_ids,
                subtoken2entity_edge_index=subtoken2entity_edge_index,
                num_entities=num_entities,
                corrupted_sentence_input=corrupted_sentence_input,
                token_labels=token_labels,
                pos_triples=pos_triples,
                neg_node_ids=neg_node_ids,
                has_edge_mask=has_edge_mask,
                link_pred_batch_type=link_pred_batch_type)

    else:
        masked_lm_loss, contrastive_loss, graph_lp_loss, intermodal_lp_loss = model.forward(
            sentence_input=sentence_input,
            token_is_entity_mask=token_is_entity_mask,
            entity_node_ids=entity_node_ids,
            subtoken2entity_edge_index=subtoken2entity_edge_index,
            num_entities=num_entities,
            corrupted_sentence_input=corrupted_sentence_input,
            token_labels=token_labels,
            pos_triples=pos_triples,
            neg_node_ids=neg_node_ids,
            has_edge_mask=has_edge_mask,
            link_pred_batch_type=link_pred_batch_type)
    total_loss = masked_lm_loss + contrastive_loss + graph_lp_loss + intermodal_lp_loss

    ld = {
        "total_loss": total_loss,
        "masked_lm_loss": masked_lm_loss,
        "contrastive_loss": contrastive_loss,
        "graph_lp_loss": graph_lp_loss,
        "intermodal_lp_loss": intermodal_lp_loss
    }
    return ld


def alignment_model_val_step(model: AlignmentModel, batch, amp, device):
    sentence_input = [t.to(device) for t in batch["sentence_input"]]
    corrupted_sentence_input = [t.to(device) for t in batch["corrupted_sentence_input"]]
    token_labels = batch["token_labels"].to(device)
    token_is_entity_mask = batch["token_is_entity_mask"].to(device)
    entity_node_ids = batch["entity_node_ids"].to(device)
    subtoken2entity_edge_index = batch["subtoken2entity_edge_index"].to(device)
    num_entities = batch["num_entities"]
    pos_triples = batch["pos_triples"].to(device)
    neg_node_ids = batch["neg_node_ids"].to(device)
    has_edge_mask = batch["has_edge_mask"].to(device)
    link_pred_batch_type = batch["batch_type"]
    model.eval()
    with torch.no_grad():
        losses_dict = model_step(model=model,
                                 sentence_input=sentence_input,
                                 token_is_entity_mask=token_is_entity_mask,
                                 entity_node_ids=entity_node_ids,
                                 subtoken2entity_edge_index=subtoken2entity_edge_index,
                                 num_entities=num_entities,
                                 corrupted_sentence_input=corrupted_sentence_input,
                                 token_labels=token_labels,
                                 pos_triples=pos_triples,
                                 neg_node_ids=neg_node_ids,
                                 has_edge_mask=has_edge_mask,
                                 link_pred_batch_type=link_pred_batch_type,
                                 amp=amp)

    return losses_dict


def alignment_model_val_epoch(model, val_loader: DataLoader, amp, device, **kwargs):
    model.eval()
    losses_dict = {"total_loss": 0., "masked_lm_loss": 0., "contrastive_loss": 0., "graph_lp_loss": 0.,
                   "intermodal_lp_loss": 0.}
    global_num_samples = 0
    for batch in val_loader:
        num_samples = len(batch["sentence_input"][0])
        step_ld = alignment_model_val_step(model=model, batch=batch, amp=amp, device=device)
        for loss_name, loss_tensor in step_ld.items():
            losses_dict[loss_name] += float(step_ld[loss_name]) * num_samples

        global_num_samples += num_samples

    for k in losses_dict.keys():
        v = losses_dict[k]
        losses_dict[k] = v / global_num_samples
    model.train()

    return losses_dict


def alignment_model_train_epoch(model, train_loader: DataLoader, val_loader: DataLoader, eval_every_n_steps: int,
                                save_chkpnt_step_interval: int, optimizer: torch.optim.Optimizer, scaler, scheduler,
                                amp, deepspeed_flag, device, step_loss_file_path, initial_global_num_steps: int,
                                **kwargs):
    output_dir = os.path.dirname(step_loss_file_path)
    model.train()
    step_losses_dict = {"total_loss": [], "masked_lm_loss": [], "contrastive_loss": [], "graph_lp_loss": [],
                        "intermodal_lp_loss": [], "learning_rate": []}
    epoch_losses_dict = {"total_loss": 0., "masked_lm_loss": 0., "contrastive_loss": 0., "graph_lp_loss": 0.,
                         "intermodal_lp_loss": 0.}

    pbar = tqdm(train_loader, miniters=len(train_loader) // 100, total=len(train_loader))
    # if init_checkpoint_step_id > 0:
    #     train_loader.dataset.return_nothing = True
    #     train_loader.dataset.skip_first_n_steps = init_checkpoint_step_id

    for in_batch_step_id, batch in enumerate(pbar):
        if len(batch) == 0:
            continue
        # current_step_id = initial_global_num_steps + in_batch_step_id
        # if current_step_id < init_checkpoint_step_id:
        #     continue
        # if current_step_id + 1 == init_checkpoint_step_id:
        #     train_loader.dataset.return_nothing = False
        #     train_loader.dataset.init_checkpoint_step_id = None
        if not deepspeed_flag:
            optimizer.zero_grad()

        step_ld = alignment_model_train_step(model=model, batch=batch, amp=amp, device=device)
        total_loss = step_ld["total_loss"]
        masked_lm_loss = step_ld['masked_lm_loss']
        contrastive_loss = step_ld['contrastive_loss']
        graph_lp_loss = step_ld['graph_lp_loss']
        intermodal_lp_loss = step_ld['intermodal_lp_loss']
        # TODO: Вот здесь для отладки надо засунуть в tqdm тип батча!!!!!!!!!!!1

        pbar.set_description(f"total_loss: {float(total_loss):.5f}. "
                             f"MLM: {float(masked_lm_loss):.5f}, "
                             f"CL: {float(contrastive_loss):.5f}, "
                             f"GLP: {float(graph_lp_loss):.5f}, "
                             f"ILP: {float(intermodal_lp_loss):.5f}. "
                             f"LR: {float(optimizer.param_groups[0]['lr']):.10f}")
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
            optimizer.zero_grad()
            logging.info(f"Evaluating at step: {initial_global_num_steps + in_batch_step_id}...")
            val_losses_dict = alignment_model_val_epoch(model, val_loader, amp, device, **kwargs)
            val_total_loss = val_losses_dict["total_loss"]
            val_mlml = val_losses_dict["masked_lm_loss"]
            val_cl = val_losses_dict["contrastive_loss"]
            val_glp = val_losses_dict["graph_lp_loss"]
            val_ilp = val_losses_dict["intermodal_lp_loss"]

            logging.info(f"Step {initial_global_num_steps + in_batch_step_id}, "
                         f"total loss: {val_total_loss}, "
                         f"MLM: {val_mlml:.5f}, "
                         f"CL: {val_cl:.5f}, "
                         f"GLP: {val_glp:.5f}, "
                         f"ILP: {val_ilp:.5f}")
        if (save_chkpnt_step_interval is not None
                and (initial_global_num_steps + in_batch_step_id) % save_chkpnt_step_interval == 0):
            logging.info(f"Saving checkpoint at step: {initial_global_num_steps + in_batch_step_id}...")
            save_model_checkpoint(model, epoch_id=(initial_global_num_steps + in_batch_step_id) // len(train_loader),
                                  num_steps=initial_global_num_steps + in_batch_step_id,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  scaler=scaler,
                                  output_dir=output_dir)
        for k, v in step_ld.items():
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

    from textkb.utils.arg_parsing import parse_modular_alignment_model_args
    args = parse_modular_alignment_model_args()
    # from textkb.utils.arg_parsing_debug import parse_modular_alignment_model_args_debug
    # args = parse_modular_alignment_model_args_debug()
    print(args)
    os.environ["MASTER_PORT"] = "8888"
    set_random_seed(args.random_seed)
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )

    use_cuda = args.use_cuda
    if use_cuda and not torch.cuda.is_available():
        raise Exception(f"Provided use_cuda flag but no CUDA GPU is available")
    # device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    contrastive_loss_name = args.contrastive_loss
    node_embs_path = args.node_embs_path

    if node_embs_path is not None:
        node_embs = np.load(node_embs_path)
        logging.info(f"Loaded node embeddings from {node_embs_path}. The size is {node_embs.shape}")
    else:
        node_embs = np.ones(shape=(700000, 128))
    num_nodes = node_embs.shape[0]

    mlm_task = args.mlm_task
    contrastive_task = args.contrastive_task
    graph_lp_task = args.graph_lp_task
    intermodal_lp_task = args.intermodal_lp_task
    freeze_node_embs = args.freeze_node_embs
    link_score_type = args.link_score_type
    link_neg_samples = args.link_negative_sample_size
    link_regularizer_weight = args.link_regularizer_weight
    entity_masking_prob = args.entity_masking_prob
    mlm_probability = args.mlm_probability
    embedding_transform = args.embedding_transform

    graph_data_dir = args.graph_data_dir
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    num_train_batches = sum(1 for e in os.listdir(train_data_dir))

    rel2id_path = os.path.join(graph_data_dir, "rel2id")
    rel2id = load_dict(rel2id_path, dtype_1=str, dtype_2=int)
    num_rels = len(rel2id)

    unique_concept_idx_path = os.path.join(graph_data_dir, "mentioned_concepts_idx")
    mentioned_concept_ids = load_list_elem_per_line(input_path=unique_concept_idx_path, dtype=int)
    global2local_concept_id = {global_id: local_id for local_id, global_id in enumerate(mentioned_concept_ids)}

    bert_encoder_name = args.bert_encoder_name
    sentence_max_length = args.sentence_max_length
    num_epochs = args.num_epochs
    save_every_n_steps = args.save_every_N_steps
    eval_every_n_steps = args.eval_every_N_steps

    deepspeed_flag = args.deepspeed
    if deepspeed_flag:
        deepspeed_cfg_path = args.deepspeed_cfg_path


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
    cot_s = f"_cot_{contrastive_loss_name}" if contrastive_task else ""
    glp_s = f"_glp_{link_score_type}" if graph_lp_task else ""
    ilp_s = f"_ilp_{link_score_type}" if intermodal_lp_task else ""
    emb_s = f"_emb_{embedding_transform}"
    freeze_nodes_s = "_frozen_node" if freeze_node_embs else ""

    output_subdir = (f"contr_l_{contrastive_loss_weight}{mlm_s}{cot_s}{glp_s}{ilp_s}{freeze_nodes_s}"
                     f"{emb_s}{freeze_nodes_s}_link_neg_{link_neg_samples}_blr_{bert_lr}_graph_lr_{graph_lr}")
    output_dir = os.path.join(output_dir, output_subdir)
    create_dir_if_not_exists(output_dir)

    bert_encoder = AutoModel.from_pretrained(bert_encoder_name)
    sentence_bert_tokenizer = AutoTokenizer.from_pretrained(bert_encoder_name)

    if contrastive_loss_name == "ms-loss":
        contrastive_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    elif contrastive_loss_name == "infonce":
        # TODO: The MoCo paper uses 0.07, while SimCLR uses 0.5.
        contrastive_loss = losses.NTXentLoss(temperature=0.07)
    else:
        raise NotImplementedError(f"unsupported contrastive loss: {contrastive_loss_name}")

    alignment_model = ModularAlignmentModel(bert_encoder=bert_encoder,
                                            bert_tokenizer=sentence_bert_tokenizer,
                                            node_embs=node_embs,
                                            contrastive_loss=contrastive_loss,
                                            multigpu=multigpu_flag,
                                            mlm_task=mlm_task,
                                            contrastive_task=contrastive_task,
                                            graph_lp_task=graph_lp_task,
                                            intermodal_lp_task=intermodal_lp_task,
                                            num_rels=num_rels,
                                            score_type=link_score_type,
                                            freeze_node_embs=freeze_node_embs,
                                            link_regularizer_weight=link_regularizer_weight,
                                            embedding_transform=embedding_transform,
                                            device=device)  # .to(device)

    # TODO: Если в директории есть config, то каким будет num_train_batches?
    # TODO: Пойти переделать subsample 20k - надо перенумеровать файлы
    train_dataset = CachedPrecomputedGraphTextDataset(tokenizer=sentence_bert_tokenizer,
                                                      data_dir=train_data_dir,
                                                      num_batches=num_train_batches,
                                                      entity_masking_prob=entity_masking_prob,
                                                      sentence_max_length=sentence_max_length,
                                                      mlm_probability=mlm_probability,
                                                      num_nodes=num_nodes,
                                                      link_negative_sample_size=link_neg_samples)

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
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=args.dataloader_num_workers,
                                      shuffle=True, collate_fn=lambda lst: lst[0])
    else:
        assert init_checkpoint_step_id > 0
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0,
                                      shuffle=True, collate_fn=lambda lst: lst[0])

    val_loader = None
    val_epoch_fn = None
    if val_data_dir is not None:
        num_val_batches = sum(1 for e in os.listdir(val_data_dir))
        val_dataset = CachedPrecomputedGraphTextDataset(tokenizer=sentence_bert_tokenizer,
                                                        data_dir=val_data_dir,
                                                        num_batches=num_val_batches,
                                                        entity_masking_prob=entity_masking_prob,
                                                        sentence_max_length=sentence_max_length,
                                                        mlm_probability=mlm_probability,
                                                        num_nodes=num_nodes,
                                                        link_negative_sample_size=link_neg_samples)

        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.dataloader_num_workers,
                                shuffle=False, collate_fn=lambda lst: lst[0])
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
    # TODO
    if deepspeed_flag:
        alignment_model, optimizer, _, scheduler = deepspeed.initialize(model=alignment_model,
                                                                     optimizer=optimizer,
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
                device=device, )

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
