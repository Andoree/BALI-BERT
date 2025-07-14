import logging
import os
import random
from argparse import ArgumentParser
from typing import Union, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from textkb.data.dataset import GraphNeighborsDataset
from textkb.gebert_models.gat_encoder import GeBertGATv2Encoder
from textkb.modeling.model import AlignmentModel
from textkb.utils.io import create_dir_if_not_exists, load_dict, load_adjacency_lists, load_tokenized_concepts, \
    load_list_elem_per_line


def load_gebert_from_checkpoint(gebert_checkpoint_path: str, add_self_loops,
                                device: Union[str, torch.device]):
    gebert_checkpoint_path = gebert_checkpoint_path
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
                                       add_self_loops=add_self_loops, layernorm_output=True, )
    logging.info(f"Loading state from GEBERT's graph encoder")
    graph_encoder.load_state_dict(gebert_checkpoint["graph_encoder"])
    del gebert_checkpoint

    return concept_bert_encoder, graph_encoder, concept_bert_encoder_name


def sample_random_edge_str(tokenizer, node_id2input_ids, input_ids, entity_node_ids, global2local_concept_id,
                           edge_index):
    mask_token_id = tokenizer.mask_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    spec_token_ids = (mask_token_id, sep_token_id, cls_token_id, pad_token_id)

    num_edges = len(edge_index)
    edge_id = random.randrange(0, num_edges)

    edge_src_node_id = edge_index[0][edge_id]
    edge_trg_node_id = edge_index[1][edge_id]

    src_node_input_ids = [x for x in input_ids[edge_src_node_id] if x not in spec_token_ids]
    trg_node_input_ids = [x for x in input_ids[edge_trg_node_id] if x not in spec_token_ids]
    src_node_tokens = tokenizer.convert_ids_to_tokens(src_node_input_ids)
    trg_node_tokens = tokenizer.convert_ids_to_tokens(trg_node_input_ids)
    src_concept_name = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in src_node_tokens))
    trg_concept_name = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in trg_node_tokens))

    # TODO: Тут что-то не так
    entity_id = entity_node_ids[edge_trg_node_id].item()
    local_entity_id = global2local_concept_id[entity_id]
    node_inp_ids = [x for x in node_id2input_ids[entity_id][0] if x not in spec_token_ids]

    node_gold_tokens = tokenizer.convert_ids_to_tokens(node_inp_ids)
    node_concept_name = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in node_gold_tokens))

    s = f"{src_concept_name} --> {trg_concept_name} ({node_concept_name})"

    return s


def gebert_infer_concept_graph_embeddings_matrix(graph_dataloader, text_graph_model, overall_num_concepts,
                                                 global2local_concept_id, node_id2input_ids, bert_tokenizer,
                                                 output_dir, device):
    hidden_size = text_graph_model.bert_hidden_dim
    concept_name_counter = np.zeros(shape=overall_num_concepts, dtype=np.int32)
    embeddings_matrix = np.zeros(shape=(overall_num_concepts, hidden_size), dtype=np.float32)
    sanity_check_log_path = os.path.join(output_dir, "edge_examples.txt")
    text_graph_model.eval()
    with open(sanity_check_log_path, 'w', encoding="utf-8") as out_file:
        for batch in tqdm(graph_dataloader, total=len(graph_dataloader), miniters=len(graph_dataloader) // 1000):
            concept_graph_input = [t.to(device) for t in batch["node_input"]]
            entity_node_ids = batch["entity_node_ids"]
            batch_num_concepts = len(entity_node_ids)
            concept_graph_edge_index = batch["concept_graph_edge_index"].to(device)

            (concept_graph_input_ids, concept_graph_att_mask) = concept_graph_input
            with torch.no_grad():
                # <num_ALL_entities, seq, h> - embedding graph concept names
                graph_concept_embs = textkb.modeling.modeling_utils.bert_encode(text_graph_model.concept_bert_encoder,
                                                                                concept_graph_input_ids,
                                                                                concept_graph_att_mask).detach()
                # <num_ALL_entitizes, h> - mean pooling bert embeddings of concept names
                # graph_concept_embs = text_graph_model.mean_pooling(graph_concept_embs, concept_graph_att_mask)
                graph_concept_embs = graph_concept_embs[:, 0]

                graph_concept_embs = text_graph_model.graph_encoder(x=graph_concept_embs,
                                                                    edge_index=concept_graph_edge_index,
                                                                    num_trg_nodes=batch_num_concepts)[
                                     :batch_num_concepts].detach()

                edge_str = sample_random_edge_str(tokenizer=bert_tokenizer,
                                                  node_id2input_ids=node_id2input_ids,
                                                  input_ids=concept_graph_input_ids,
                                                  entity_node_ids=entity_node_ids,
                                                  global2local_concept_id=global2local_concept_id,
                                                  edge_index=concept_graph_edge_index)
                out_file.write(f"{edge_str}\n")

                assert graph_concept_embs.size(0) == len(entity_node_ids)

                for global_concept_id, emb in zip(entity_node_ids, graph_concept_embs):
                    global_concept_id = global_concept_id.item()
                    local_concept_id = global2local_concept_id[global_concept_id]
                    concept_name_counter[local_concept_id] += 1
                    embeddings_matrix[local_concept_id] += emb.detach().cpu().numpy()
    concept_name_counter = concept_name_counter.reshape((overall_num_concepts, 1))
    embeddings_matrix = embeddings_matrix / concept_name_counter

    return embeddings_matrix


def main(args):
    gebert_checkpoint_path = args.gebert_checkpoint_path
    concept_max_length = args.concept_max_length
    use_cuda = args.use_cuda
    use_rel = args.use_rel
    max_n_neighbors = args.max_n_neighbors
    masking_mode = args.masking_mode

    tokenized_concepts_path = args.tokenized_concepts_path
    graph_data_dir = args.graph_data_dir
    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")
    batch_size = args.batch_size
    dataloader_num_workers = args.dataloader_num_workers

    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    create_dir_if_not_exists(output_dir)

    # bert_tokenizer = AutoTokenizer.from_pretrained(bert_encoder_name)

    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")

    node_id2input_ids: List[Tuple[Tuple[int]]] = load_tokenized_concepts(tok_conc_path=tokenized_concepts_path)
    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, use_rel)
    mentioned_concepts_idx_path = os.path.join(graph_data_dir, "mentioned_concepts_idx")
    mentioned_concept_ids = load_list_elem_per_line(input_path=mentioned_concepts_idx_path, dtype=int)
    global2local_concept_id = {global_id: local_id for local_id, global_id in enumerate(mentioned_concept_ids)}

    concept_bert_encoder, graph_encoder, bert_encoder_name = load_gebert_from_checkpoint(gebert_checkpoint_path,
                                                                                         add_self_loops=True,
                                                                                         device=device)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_encoder_name)
    node_input_ids = []
    node_idx = []
    for idx in mentioned_concept_ids:
        inp_ids = node_id2input_ids[idx]
        num_synonyms = len(inp_ids)
        # tokens = [bert_tokenizer.convert_ids_to_tokens(n_ids) for n_ids in inp_ids]
        # tokens = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in tokens]
        # print(idx, "||".join(tokens))
        # tokens = [bert_tokenizer.convert_ids_to_tokens(n_ids) for n_ids in node_id2input_ids[idx]]
        # tokens = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in tokens]
        # print(idx, "||".join(tokens))
        # print('--')
        node_input_ids.extend(inp_ids)
        node_idx.extend([idx, ] * num_synonyms)
    assert len(node_input_ids) == len(node_idx)

    logging.info(f"Flattened node input. There are {len(node_input_ids)} unique concept names.")
    graph_dataset = GraphNeighborsDataset(tokenizer=bert_tokenizer,
                                          node_id2adjacency_list=node_id2adjacency_list,
                                          node_input_ids=node_input_ids,
                                          node_id2input_ids=node_id2input_ids,
                                          max_n_neighbors=max_n_neighbors,
                                          use_rel=use_rel,
                                          masking_mode=masking_mode,
                                          concept_max_length=concept_max_length,
                                          central_node_idx=node_idx)
    graph_dataloader = DataLoader(graph_dataset, batch_size=batch_size,
                                  num_workers=dataloader_num_workers,
                                  shuffle=False,
                                  collate_fn=graph_dataset.collate_fn)
    # TODO: КАКОЕ У МЕНЯ ВСЁ-ТАКИ МАКСИРОВАНИЕ?

    alignment_model = AlignmentModel(sentence_bert_encoder=concept_bert_encoder,
                                     concept_bert_encoder=concept_bert_encoder,
                                     graph_encoder=graph_encoder,
                                     contrastive_loss=None,
                                     multigpu=False,
                                     freeze_graph_bert_encoder=True,
                                     freeze_graph_encoder=True).to(device)

    embeddings_matrix = gebert_infer_concept_graph_embeddings_matrix(graph_dataloader,
                                                                     text_graph_model=alignment_model,
                                                                     overall_num_concepts=len(mentioned_concept_ids),
                                                                     global2local_concept_id=global2local_concept_id,
                                                                     node_id2input_ids=node_id2input_ids,
                                                                     bert_tokenizer=bert_tokenizer,
                                                                     output_dir=output_dir,
                                                                     device=device)
    # output_embs_path = os.path.join(output_dir, "node_embs_gebert.npy")
    np.save(output_path, embeddings_matrix)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    # parser.add_argument("--", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/tokenized_sentences")
    # parser.add_argument("--output_dir", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/numpy_tokenized_sentences")
    parser.add_argument("--tokenized_concepts_path", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--gebert_checkpoint_path", required=False,
                        default="/home/c204/University/NLP/BERN2_sample/gebert_checkpoint_for_debug/checkpoint_e_1_steps_94765.pth")
    parser.add_argument("--graph_data_dir", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug")

    parser.add_argument("--use_cuda", action="store_true", )
    parser.add_argument("--masking_mode", choices=("text", "graph", "both", "random"), type=str,
                        default="random")
    parser.add_argument("--concept_max_length", type=int, default=32)
    parser.add_argument("--max_n_neighbors", type=int, default=1)
    parser.add_argument("--use_rel", action="store_true", )
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--output_path", type=str, default="DELETE/")

    args = parser.parse_args()
    main(args)
