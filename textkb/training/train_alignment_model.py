import argparse
import logging
import os.path
import time
from typing import Tuple, List

import pandas as pd
import torch.cuda
from pytorch_metric_learning import losses
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from textkb.data.dataset import TextGraphGraphNeighborsDataset
from textkb.gebert_models.gat_encoder import GeBertGATv2Encoder
from textkb.modeling.graph_encoders import GATv2Encoder
from textkb.modeling.model import AlignmentModel
from textkb.training.training import train_model
from textkb.utils.io import load_adjacency_lists, load_tokenized_sentences_data, load_tokenized_concepts, \
    create_dir_if_not_exists, save_dict, load_dict, load_tokenized_sentences_data_v2
from textkb.utils.model_utils import save_model_checkpoint
from textkb.utils.utils import validate_sentence_concept_tokenization


def alignment_model_train_step(model: AlignmentModel, batch, amp, device):
    sentence_input = [t.to(device) for t in batch["sentence_input"]]
    concept_graph_input = [t.to(device) for t in batch["node_input"]]
    # print('batch["entity_node_ids"]', batch["entity_node_ids"])
    entity_node_ids = batch["entity_node_ids"].to(device)

    token_is_entity_mask = batch["token_is_entity_mask"]
    subtoken2entity_edge_index = batch["subtoken2entity_edge_index"].to(device)
    concept_graph_edge_index = batch["concept_graph_edge_index"].to(device)
    num_entities = batch["num_entities"]
    contrastive_loss = model_step(model, sentence_input, concept_graph_input, entity_node_ids, token_is_entity_mask,
                                  subtoken2entity_edge_index, concept_graph_edge_index, num_entities, amp)

    return contrastive_loss


def model_step(model, sentence_input, concept_graph_input, entity_node_ids, token_is_entity_mask,
               subtoken2entity_edge_index, concept_graph_edge_index, num_entities, amp):
    if amp:
        with autocast():
            contrastive_loss = model.forward(sentence_input=sentence_input,
                                             concept_graph_input=concept_graph_input,
                                             entity_node_ids=entity_node_ids,
                                             token_is_entity_mask=token_is_entity_mask,
                                             subtoken2entity_edge_index=subtoken2entity_edge_index,
                                             concept_graph_edge_index=concept_graph_edge_index,
                                             num_entities=num_entities)

    else:
        contrastive_loss = model.forward(sentence_input=sentence_input,
                                         concept_graph_input=concept_graph_input,
                                         entity_node_ids=entity_node_ids,
                                         token_is_entity_mask=token_is_entity_mask,
                                         subtoken2entity_edge_index=subtoken2entity_edge_index,
                                         concept_graph_edge_index=concept_graph_edge_index,
                                         num_entities=num_entities)
    return contrastive_loss


def alignment_model_val_step(model: AlignmentModel, batch, amp, device):
    sentence_input = [t.to(device) for t in batch["sentence_input"]]
    concept_graph_input = [t.to(device) for t in batch["node_input"]]
    entity_node_ids = batch["entity_node_ids"].to(device)

    token_is_entity_mask = batch["token_is_entity_mask"]
    subtoken2entity_edge_index = batch["subtoken2entity_edge_index"].to(device)
    concept_graph_edge_index = batch["concept_graph_edge_index"].to(device)
    num_entities = batch["num_entities"]
    model.eval()
    with torch.no_grad():
        contrastive_loss = model_step(model, sentence_input, concept_graph_input, entity_node_ids, token_is_entity_mask,
                                      subtoken2entity_edge_index, concept_graph_edge_index, num_entities, amp)
    # logging.info(f"Train loss: {float(loss)}")
    return contrastive_loss


def alignment_model_val_epoch(model, val_loader: DataLoader, optimizer: torch.optim.Optimizer, amp, device, **kwargs):
    model.eval()
    losses_dict = {"contrastive_loss": 0., }
    global_num_samples = 0
    for batch in val_loader:
        num_samples = len(batch["sentence_input"][0])
        optimizer.zero_grad()
        contrastive_loss = alignment_model_val_step(model=model, batch=batch, amp=amp, device=device)
        global_num_samples += num_samples
        losses_dict["contrastive_loss"] += float(contrastive_loss) * num_samples
    for k in losses_dict.keys():
        v = losses_dict[k]
        losses_dict[k] = v / global_num_samples
    model.train()

    return losses_dict["contrastive_loss"]


def alignment_model_train_epoch(model, train_loader: DataLoader, val_loader: DataLoader, eval_every_n_steps: int,
                                save_chkpnt_step_interval: int, optimizer: torch.optim.Optimizer, scaler, scheduler,
                                amp, device, step_loss_file_path, initial_global_num_steps: int, **kwargs):
    output_dir = os.path.dirname(step_loss_file_path)
    model.train()
    step_losses_dict = {"contrastive_loss": [], "learning_rate": []}
    losses_dict = {"contrastive_loss": 0., }
    num_steps = 0
    pbar = tqdm(train_loader, miniters=len(train_loader) // 100, total=len(train_loader))
    for batch in pbar:
        optimizer.zero_grad()
        contrastive_loss = alignment_model_train_step(model=model, batch=batch, amp=amp, device=device)
        pbar.set_description(f"Loss: {float(contrastive_loss):.5f}")
        if amp:
            scaler.scale(contrastive_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            contrastive_loss.backward()
            optimizer.step()
        scheduler.step()
        step_losses_dict["learning_rate"].append(optimizer.param_groups[0]["lr"])
        num_steps += 1

        if eval_every_n_steps is not None and (initial_global_num_steps + num_steps) % eval_every_n_steps == 0:
            logging.info(f"Evaluating at step: {initial_global_num_steps + num_steps}...")
            val_contrastive_loss = alignment_model_val_epoch(model, val_loader, optimizer, amp, device, **kwargs)
            logging.info(f"Step {initial_global_num_steps + num_steps}, loss: {val_contrastive_loss}")
        if (save_chkpnt_step_interval is not None
                and (initial_global_num_steps + num_steps) % save_chkpnt_step_interval == 0):
            logging.info(f"Saving checkpoint at step: {initial_global_num_steps + num_steps}...")
            save_model_checkpoint(model, epoch_id=-1,
                                  num_steps=initial_global_num_steps + num_steps,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  output_dir=output_dir)
        losses_dict["contrastive_loss"] += float(contrastive_loss)
        step_losses_dict["contrastive_loss"].append(float(contrastive_loss))
    epoch_losses_df = pd.DataFrame(step_losses_dict)
    epoch_losses_df.to_csv(step_loss_file_path, sep='\t')

    losses_dict = {key: lo / (num_steps + 1e-9) for key, lo in losses_dict.items()}

    return losses_dict["contrastive_loss"], num_steps


def main():
    # TODO: ОБНОВИТЬ SUB-ДИРЕКТОРИЮ!!
    # TODO: В основной скрит добавить проверку на то, что do_lower_case одинаковый для токенизации concept_names
    # TODO: и предложений!!!!
    # from textkb.utils.arg_parsing import parse_alignment_model_args
    # args = parse_alignment_model_args(graph_encoder_name="GAT")
    from textkb.utils.arg_parsing_debug import parse_alignment_model_args_debug
    args = parse_alignment_model_args_debug(graph_encoder_name="GAT")
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)
    model_descr_path = os.path.join(output_dir, "model_description.tsv")
    save_dict(save_path=model_descr_path, dictionary=vars(args), )

    use_cuda = args.use_cuda
    if use_cuda and not torch.cuda.is_available():
        raise Exception(f"Provided use_cuda flag but no CUDA GPU is available")

    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    contrastive_loss_name = args.contrastive_loss
    graph_data_dir = args.graph_data_dir
    bert_encoder_name = args.bert_encoder_name
    graph_model_type = args.graph_model_type
    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")

    max_n_neighbors = args.max_n_neighbors
    sentence_max_length = args.sentence_max_length
    concept_max_length = args.concept_max_length
    num_epochs = args.num_epochs
    save_every_n_steps = args.save_every_N_steps
    eval_every_n_steps = args.eval_every_N_steps

    amp_flag = args.amp
    multigpu_flag = args.parallel
    save_every_N_epoch = args.save_every_N_epoch
    bert_lr = args.bert_learning_rate
    graph_lr = args.graph_learning_rate
    max_num_warmup_steps = args.max_num_warmup_steps
    warmup_steps_ratio = args.warmup_steps_ratio
    weight_decay = args.weight_decay
    masking_mode = args.masking_mode

    train_tokenized_sentences_path = args.train_tokenized_sentences_path
    val_tokenized_sentences_path = args.val_tokenized_sentences_path
    tokenized_concepts_path = args.tokenized_concepts_path

    gat_num_layers = args.gat_num_layers
    gat_num_hidden_channels = args.gat_num_hidden_channels
    gat_dropout_p = args.gat_dropout_p
    gat_num_att_heads = args.gat_num_att_heads
    gat_attention_dropout_p = args.gat_attention_dropout_p
    gat_add_self_loops = args.gat_add_self_loops

    use_rel = args.use_rel
    freeze_graph_bert_encoder = args.freeze_graph_bert_encoder
    freeze_graph_encoder = args.freeze_graph_encoder
    if freeze_graph_bert_encoder or freeze_graph_encoder:
        assert graph_model_type not in ("shared_encoder",)
    remove_selfloops = args.remove_selfloops
    if remove_selfloops:
        raise NotImplementedError(f"remove_selfloops: {remove_selfloops}")
    contrastive_loss_weight = args.contrastive_loss_weight
    batch_size = args.batch_size
    model_checkpoint_path = args.model_checkpoint_path

    validate_sentence_concept_tokenization(train_tokenized_sentences_path, tokenized_concepts_path)
    if val_tokenized_sentences_path is not None:
        validate_sentence_concept_tokenization(val_tokenized_sentences_path, tokenized_concepts_path)
    graph_model_type_string = graph_model_type[:3]
    fr_bert_s = "_fr_bert" if freeze_graph_bert_encoder else ""
    fr_graph_s = "_fr_graph" if freeze_graph_encoder else ""
    output_subdir = f"gat_{gat_num_layers}_{gat_num_hidden_channels}_{gat_dropout_p}_" \
                    f"{gat_num_att_heads}_{gat_attention_dropout_p}_add_loops_{gat_add_self_loops}" \
                    f"_contr_l_{contrastive_loss_weight}_use_rel_{use_rel}_rl_{remove_selfloops}" \
                    f"_{graph_model_type_string}{fr_bert_s}{fr_graph_s}_blr_{bert_lr}_graph_lr_{graph_lr}_b_{batch_size}"
    output_dir = os.path.join(output_dir, output_subdir)
    create_dir_if_not_exists(output_dir)

    sentence_bert_encoder = AutoModel.from_pretrained(bert_encoder_name)
    sentence_bert_tokenizer = AutoTokenizer.from_pretrained(bert_encoder_name)
    bert_hidden_size = sentence_bert_encoder.config.hidden_size

    if contrastive_loss_name == "ms-loss":
        contrastive_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    elif contrastive_loss_name == "infonce":
        # TODO: The MoCo paper uses 0.07, while SimCLR uses 0.5.
        contrastive_loss = losses.NTXentLoss(temperature=0.07)
    else:
        raise NotImplementedError(f"unsupported contrastive loss: {contrastive_loss_name}")

    if graph_model_type == "shared_encoder":
        graph_encoder = GATv2Encoder(in_channels=bert_hidden_size, num_layers=gat_num_layers,
                                     num_hidden_channels=gat_num_hidden_channels,
                                     dropout_p=gat_dropout_p,
                                     num_att_heads=gat_num_att_heads,
                                     attention_dropout_p=gat_attention_dropout_p,
                                     add_self_loops=gat_add_self_loops, multigpu=False)
        concept_bert_encoder = sentence_bert_encoder

    elif graph_model_type == "gebert":
        gebert_checkpoint_path = args.gebert_checkpoint_path
        gebert_dir = os.path.dirname(gebert_checkpoint_path)
        gebert_config_path = os.path.join(gebert_dir, "model_description.tsv")
        gebert_config = load_dict(gebert_config_path, sep='\t')
        gebert_num_inner_layers = int(gebert_config["gat_num_inner_layers"])
        gebert_num_hidden_channels = int(gebert_config["gat_num_hidden_channels"])
        gebert_num_att_heads = int(gebert_config["gat_num_att_heads"])
        gebert_dropout_p = float(gebert_config["gat_dropout_p"])
        gebert_attention_dropout_p = float(gebert_config["gat_attention_dropout_p"])
        concept_bert_encoder_name = gebert_config["text_encoder"]
        gebert_checkpoint = torch.load(gebert_checkpoint_path, map_location=device)

        concept_bert_encoder = AutoModel.from_pretrained(concept_bert_encoder_name)
        logging.info(f"Loading state from GEBERT's BERT encoder")
        concept_bert_encoder.load_state_dict(gebert_checkpoint["model_state"])
        logging.info(f"Using pre-trained GEBERT encoder with parameters:")
        logging.info(f"\tnum layers: {gebert_num_inner_layers}")
        logging.info(f"\tnum hidden_channels: {gebert_num_hidden_channels}")
        logging.info(f"\tnum num_att_heads: {gebert_num_att_heads}")
        logging.info(f"\tnum dropout_p: {gebert_dropout_p}")
        logging.info(f"\tnum attention_dropout_p: {gebert_attention_dropout_p}")
        logging.info(f"\tnum concept_bert_encoder_name: {concept_bert_encoder_name}")
        gebert_hidden_size = concept_bert_encoder.config.hidden_size

        graph_encoder = GeBertGATv2Encoder(in_channels=gebert_hidden_size, num_outer_layers=1,
                                           num_inner_layers=gebert_num_inner_layers,
                                           num_hidden_channels=gebert_num_hidden_channels, dropout_p=gebert_dropout_p,
                                           num_att_heads=gebert_num_att_heads,
                                           attention_dropout_p=gebert_attention_dropout_p,
                                           add_self_loops=gat_add_self_loops, layernorm_output=True,)

        logging.info(f"Loading state from GEBERT's graph encoder")
        graph_encoder.load_state_dict(gebert_checkpoint["graph_encoder"])
        del gebert_checkpoint

    else:
        raise ValueError(f"Invalid graph_model_type: {graph_model_type}")

    alignment_model = AlignmentModel(sentence_bert_encoder=sentence_bert_encoder,
                                     concept_bert_encoder=concept_bert_encoder,
                                     graph_encoder=graph_encoder,
                                     contrastive_loss=contrastive_loss,
                                     multigpu=multigpu_flag,
                                     freeze_graph_bert_encoder=freeze_graph_bert_encoder,
                                     freeze_graph_encoder=freeze_graph_encoder).to(device)
    if freeze_graph_encoder:
        logging.info(f"Using freezed GEBERT's graph encoder...")
        for param in alignment_model.graph_encoder.parameters():
            param.requires_grad = False
    if freeze_graph_bert_encoder:
        logging.info(f"Freezing graph modality's BERT encoder...")
        for param in alignment_model.concept_bert_encoder.parameters():
            param.requires_grad = False

    # tr_tokenized_data_dict = load_tokenized_sentences_data(tokenized_data_dir=train_tokenized_sentences_path)
    tr_tokenized_data_dict = load_tokenized_sentences_data_v2(tokenized_data_dir=train_tokenized_sentences_path)
    node_id2input_ids: List[Tuple[Tuple[int]]] = load_tokenized_concepts(tok_conc_path=tokenized_concepts_path)
    # tr_tokenized_data_dict = load_tokenized_sentences_data(tokenized_data_dir=train_tokenized_sentences_path)

    # tr_tokenized_data_dict = load_tokenized_sentences_data_parallel(tokenized_data_dir=train_tokenized_sentences_path,
    #                                                                 n_proc=4)
    tr_sent_input_ids: List[Tuple[int]] = tr_tokenized_data_dict["input_ids"]
    tr_token_ent_b_masks: List[Tuple[int]] = tr_tokenized_data_dict["token_entity_mask"]
    tr_edge_index_token_idx: List[Tuple[int]] = tr_tokenized_data_dict["edge_index_token_idx"]
    tr_edge_index_entity_idx: List[Tuple[int]] = tr_tokenized_data_dict["edge_index_entity_idx"]

    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, use_rel)

    train_dataset = TextGraphGraphNeighborsDataset(tokenizer=sentence_bert_tokenizer,
                                                   sentence_input_ids=tr_sent_input_ids,
                                                   token_ent_binary_masks=tr_token_ent_b_masks,
                                                   edge_index_token_idx=tr_edge_index_token_idx,
                                                   edge_index_entity_idx=tr_edge_index_entity_idx,
                                                   node_id2adjacency_list=node_id2adjacency_list,
                                                   node_id2input_ids=node_id2input_ids,
                                                   max_n_neighbors=max_n_neighbors,
                                                   use_rel=use_rel,
                                                   masking_mode=masking_mode,
                                                   sentence_max_length=sentence_max_length,
                                                   concept_max_length=concept_max_length)
    val_loader = None
    val_epoch_fn = None
    if val_tokenized_sentences_path is not None:
        val_tokenized_data_dict = load_tokenized_sentences_data_v2(tokenized_data_dir=val_tokenized_sentences_path)

        val_sent_input_ids: List[Tuple[int]] = val_tokenized_data_dict["input_ids"]
        val_token_ent_b_masks: List[Tuple[int]] = val_tokenized_data_dict["token_entity_mask"]
        val_edge_index_token_idx: List[Tuple[int]] = val_tokenized_data_dict["edge_index_token_idx"]
        val_edge_index_entity_idx: List[Tuple[int]] = val_tokenized_data_dict["edge_index_entity_idx"]

        val_dataset = TextGraphGraphNeighborsDataset(tokenizer=sentence_bert_tokenizer,
                                                     sentence_input_ids=val_sent_input_ids,
                                                     token_ent_binary_masks=val_token_ent_b_masks,
                                                     edge_index_token_idx=val_edge_index_token_idx,
                                                     edge_index_entity_idx=val_edge_index_entity_idx,
                                                     node_id2adjacency_list=node_id2adjacency_list,
                                                     node_id2input_ids=node_id2input_ids,
                                                     max_n_neighbors=max_n_neighbors,
                                                     use_rel=use_rel,
                                                     masking_mode=masking_mode,
                                                     sentence_max_length=sentence_max_length,
                                                     concept_max_length=concept_max_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.dataloader_num_workers,
                                shuffle=False, collate_fn=val_dataset.collate_fn)
        val_epoch_fn = alignment_model_val_epoch
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.dataloader_num_workers,
                                  shuffle=True, collate_fn=train_dataset.collate_fn)

    logging.info(f"Using {bert_lr} LR for BERT and {graph_lr} for graph encoder")
    opt_params = [{"params": alignment_model.graph_encoder.parameters(), "lr": graph_lr},
                  {"params": alignment_model.sentence_bert_encoder.parameters(), "lr": bert_lr},
                  ]
    if alignment_model.sentence_bert_encoder != alignment_model.concept_bert_encoder and graph_model_type != "shared_encoder":
        opt_params.append({"params": alignment_model.concept_bert_encoder.parameters(), "lr": bert_lr})
    optimizer = torch.optim.AdamW(opt_params,
                                  weight_decay=weight_decay)
    num_batches = len(train_dataloader)
    overall_num_steps = num_batches * num_epochs
    num_warmup_steps = min(int(warmup_steps_ratio * overall_num_steps), max_num_warmup_steps)
    logging.info(f"LR scheduler parameters: {num_warmup_steps} warm-up steps, {overall_num_steps} total num steps")
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=overall_num_steps)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    train_model(model=alignment_model, optimizer=optimizer, train_epoch_fn=alignment_model_train_epoch,
                val_epoch_fn=val_epoch_fn, chkpnt_path=model_checkpoint_path, train_loader=train_dataloader,
                val_loader=val_loader, num_epochs=num_epochs, output_dir=output_dir,
                contrastive_loss_weight=contrastive_loss_weight, scaler=scaler, scheduler=scheduler,
                save_chkpnt_epoch_interval=save_every_N_epoch, save_chkpnt_step_interval=save_every_n_steps,
                eval_every_n_steps=eval_every_n_steps, amp=amp_flag, device=device, )

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
