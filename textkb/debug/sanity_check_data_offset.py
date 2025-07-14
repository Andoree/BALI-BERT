import logging
import os
import random
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from textkb.data.offsets_dataset import TextGraphGraphNeighborsOffsetDataset
from textkb.utils.io import load_list_elem_per_line, \
    load_tokenized_concepts, load_offset_index, load_adjacency_lists, load_dict, create_dir_if_not_exists


def validate_attention_mask(input_ids, att_mask, tokenizer):
    assert att_mask.dim() == 2
    assert input_ids.size() == att_mask.size()
    for i, inp_ids in enumerate(input_ids):
        mask_token_id = tokenizer.mask_token_id
        pad_token_id = tokenizer.pad_token_id
        spec_toks = (mask_token_id, pad_token_id)
        # print(f"INPUT IDS VALIDATION BEFORE [MASK] DROP {len(inp_ids)}",
        #       tokenizer.convert_ids_to_tokens(inp_ids, skip_special_tokens=False))
        inp_ids = tuple(map(lambda x: 128 if x.item() == mask_token_id else x.item(), inp_ids))
        # inp_ids = tuple(filter(lambda x: x.item() not in spec_toks,  inp_ids))
        # print(f"INPUT IDS VALIDATION AFTER [MASK] DROP {len(inp_ids)}",
        #       tokenizer.convert_ids_to_tokens(inp_ids, skip_special_tokens=False))
        corr_tokens_no_spec_tokens = tokenizer.convert_ids_to_tokens(inp_ids,
                                                                     skip_special_tokens=False)
        node_name_before = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in corr_tokens_no_spec_tokens))
        corr_tokens_no_spec_tokens = tokenizer.convert_ids_to_tokens(inp_ids,
                                                                     skip_special_tokens=True)
        # TODO

        # print("tokenizer.convert_ids_to_tokens(inp_ids, skip_special_tokens=False)",
        #       tokenizer.convert_ids_to_tokens(inp_ids, skip_special_tokens=False))
        # print("corr_tokens_no_spec_tokens", corr_tokens_no_spec_tokens)
        # print("att_mask", att_mask)
        # print("len(corr_tokens_no_spec_tokens)", len(corr_tokens_no_spec_tokens))
        # print("torch.sum(att_mask[i]).item()",  torch.sum(att_mask[i]).item())

        if len(corr_tokens_no_spec_tokens) + 2 != torch.sum(att_mask[i]).item():
            logging.info(">>> ERROR???")
            node_name_after = "".join(
                (x.strip("#") if x.startswith("#") else f" {x}" for x in corr_tokens_no_spec_tokens))
            logging.info(f"\tBEFORE: {node_name_before}")
            logging.info(f"\tAFTER: {node_name_after}")
            logging.info(f"\t{corr_tokens_no_spec_tokens}")
            logging.info(f"\t{att_mask[i]}")
            logging.info(">>> ERROR???")
        # assert len(corr_tokens_no_spec_tokens) + 2 == torch.sum(att_mask[i]).item()


def offset_data_sanity_check(data_loader, tokenizer, id2rel, node_id2tokenized_terms, log_file_path):
    pbar = tqdm(data_loader, miniters=len(data_loader) // 100, total=len(data_loader))
    with open(log_file_path, 'w', encoding="utf-8") as out_file:
        for in_batch_step_id, batch in enumerate(pbar):
            if in_batch_step_id > 25:
                break
            check_strs_list = []
            corr_input_ids, corr_att_mask = batch["corrupted_sentence_input"]
            token_labels = batch["token_labels"]

            entity_node_ids = batch["entity_node_ids"]
            # token_is_entity_mask = batch["token_is_entity_mask"]
            token_is_entity_mask = None
            if batch.get("token_is_entity_mask") is not None:
                token_is_entity_mask = batch["token_is_entity_mask"]
            subtoken2entity_edge_index = None
            if batch.get("subtoken2entity_edge_index") is not None:
                subtoken2entity_edge_index = batch["subtoken2entity_edge_index"]
            entity_index_input = None
            if batch.get("entity_index_input") is not None:
                entity_index_input = batch["entity_index_input"]
                # print("entity_index_input[0]", entity_index_input[0])
                # print("entity_index_input[1]", entity_index_input[1])
                # print("entity_index_input[2]", entity_index_input[2])
            # subtoken2entity_edge_index = batch["subtoken2entity_edge_index"]
            concept_graph_edge_index = None
            if batch.get("concept_graph_edge_index") is not None:
                concept_graph_edge_index = batch["concept_graph_edge_index"]
            concept_input_ids, concept_att_mask = None, None
            if batch.get("node_input") is not None:
                concept_input_ids, concept_att_mask = batch["node_input"]

            lin_graph_input = None
            if batch.get("lin_graph_input") is not None:
                lin_graph_input = batch["lin_graph_input"]
            num_entities = batch["num_entities"]
            rel_idx = batch["rel_idx"]
            out_file.write(f"num_entities: {num_entities}\n")

            # print("corr_input_ids", corr_input_ids)
            # print('--')
            # print("token_labels", token_labels)
            corrupted_tokens = [tokenizer.convert_ids_to_tokens(t) for t in corr_input_ids]
            # [val for sublist in matrix for val in sublist]
            # print("corr_input_ids[0][0]", corr_input_ids[0][0])
            # print("tokenizer.convert_ids_to_tokens(corr_input_ids[0][0])",
            #       tokenizer.convert_ids_to_tokens(corr_input_ids[0][0].item()))
            # token_labels_strs =
            # token_labels_strs = [tokenizer.convert_ids_to_tokens(tt.item()) if tt.item() != -100 else tokenizer.convert_ids_to_tokens(
            #         corr_input_ids[i][j].item()) for i, t in enumerate(token_labels) for j, tt in enumerate(t)]
            token_labels_strs = [
                [tokenizer.convert_ids_to_tokens(tt.item()) if tt.item() != -100 else tokenizer.convert_ids_to_tokens(
                    corr_input_ids[i][j].item()) for j, tt in enumerate(t)] for i, t in enumerate(token_labels)]
            # print("token_labels_strs", token_labels_strs)

            # tokenizer.convert_ids_to_tokens(tt) if tt.item() != -100 else tokenizer.convert_ids_to_tokens(
            # corr_input_ids[i][j]) for j, tt in enumerate(t) for i, t in enumerate(token_labels)]

            # [tokenizer.convert_ids_to_tokens(t) for t in corr_input_ids]
            check_strs_list.append("Sentences:")
            for i, (corr_t, t_lab) in enumerate(zip(corrupted_tokens, token_labels_strs)):
                ######################################################
                # Validate token labels against corrupted input_ids  #
                ######################################################
                corr_sentence = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in corr_t))
                true_sentence = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t_lab))
                check_strs_list.append(f"Masked: {corr_sentence}\nUnmasked: {true_sentence}")
            # ###########################################################
            # ### Validate corrupted input_ids against attention mask ###
            # ###########################################################
            validate_attention_mask(corr_input_ids, corr_att_mask, tokenizer)
            # ####################################################
            # ### Validate token labels against attention mask ###
            # ####################################################
            # validate_attention_mask(token_labels, corr_att_mask, tokenizer)
            # ####################################################
            # Validate concept_input_ids against attention mask ##
            # ####################################################
            if concept_input_ids is not None:
                validate_attention_mask(concept_input_ids, concept_att_mask, tokenizer)
                node_tokens = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True) for t in concept_input_ids]
                # node_tokens = tokenizer.convert_ids_to_tokens(concept_input_ids)
                node_names = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
                              for t in node_tokens]

            # #######################################
            # Validate entity token -> concept name #
            # #######################################
            if token_is_entity_mask is not None:
                # print(token_is_entity_mask)
                entity_input_ids = corr_input_ids[token_is_entity_mask > 0]
                assert entity_input_ids.dim() == 1
            # central_node_input_ids = concept_input_ids
            if lin_graph_input is not None:
                lin_graph_input_ids, lin_graph_att_mask = lin_graph_input
                lin_graph_tokens = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=False)
                                    for t in lin_graph_input_ids]
                node_names = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
                              for t in lin_graph_tokens]
                assert len(node_names) == len(entity_node_ids)
                # out_file.write("Linearized Graphs:\n")
                check_strs_list.append("Linearized Graphs")
                for ln, ling_entity_id in zip(node_names, entity_node_ids):
                    # out_file.write(f"(CU.I: {ling_entity_id}) {ln}")
                    ling_entity = random.choice(node_id2tokenized_terms[ling_entity_id])
                    ling_entity = tokenizer.convert_ids_to_tokens(ling_entity, skip_special_tokens=False)
                    ling_entity = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in ling_entity))
                    check_strs_list.append(f"\t(CU.I: {ling_entity_id} ({ling_entity})) {ln}")

            # central_node_names = node_names[:num_entities]
            # central_node_tokens = node_tokens[:num_entities]
            # central_node_names = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
            #                       for t in central_node_tokens]
            if subtoken2entity_edge_index is not None:
                token_idx = subtoken2entity_edge_index[0]
                entity_idx = subtoken2entity_edge_index[1]
                # print(f"entity_idx.max() {entity_idx.max()}")
                # print(f"num_entities {num_entities}")
                # out_file.write(f"entity_idx {entity_idx.min()} {entity_idx.max()}\n")
                # out_file.write(f"token_idx {token_idx.min()} {token_idx.max()}\n")
                check_strs_list.append(f"entity_idx {len(entity_idx)} {entity_idx.min()} {entity_idx.max()}")
                check_strs_list.append(f"token_idx {len(token_idx)} {token_idx.min()} {token_idx.max()}")
                assert entity_idx.max() + 1 == num_entities
                check_strs_list.append("Subtoken2entity edges:")
                for t_id, e_id in zip(token_idx, entity_idx):
                    # TODO: У меня entity ведь вытянуты в одномерный массив? А точно ли так в самой модели??
                    token_str = tokenizer.convert_ids_to_tokens(entity_input_ids[t_id].item(), skip_special_tokens=True)
                    node_str = node_names[e_id]
                    local_entity_id = entity_node_ids[e_id]
                    # TODO
                    s = f"\t{token_str} --> {node_str} (local_id: {e_id}, {local_entity_id})"
                    check_strs_list.append(s)
            if entity_index_input is not None:
                entity_index_matrix, entity_matrix_mask, sentence_index = entity_index_input
                num_ents, max_num_tokens_in_entity = entity_index_matrix.size()
                sentence_index = sentence_index.view(-1)
                entity_index_matrix = entity_index_matrix.view(-1)
                # print("sentence_index", sentence_index.size())
                # print("entity_index_matrix", entity_index_matrix.size())

                entity_token_input_ids = corr_input_ids[list(sentence_index.view(-1)),
                list(entity_index_matrix.view(-1))]
                # print("entity_token_input_ids", entity_token_input_ids.size())
                # print("entity_index_matrix", len(entity_index_matrix))
                # print("entity_matrix_mask", len(entity_matrix_mask))
                # print("sentence_index", len(sentence_index))
                # print('//')
                assert len(entity_token_input_ids) == len(sentence_index) == len(entity_index_matrix)
                entity_token_input_ids = entity_token_input_ids.view((num_entities, max_num_tokens_in_entity))

                entity_tokens = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True)
                                 for t in entity_token_input_ids]

                entity_names = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
                                for t in entity_tokens]
                # out_file.write("Entities:\n")
                # print(f"ENTITIES, {len(entity_names)}")
                check_strs_list.append("Entities:")
                assert len(entity_node_ids) == len(entity_names) == len(node_names[:len(entity_names)])
                for ent_id, e_name, node_name in zip(entity_node_ids, entity_names, node_names[:len(entity_names)]):
                    ling_entity = random.choice(node_id2tokenized_terms[ent_id.item()])
                    ling_entity = tokenizer.convert_ids_to_tokens(ling_entity, skip_special_tokens=True)
                    ling_entity = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in ling_entity))
                    # check_strs_list.append(f"\t(CU.I: {ling_entity_id} ({ling_entity})) {ln}")

                    check_strs_list.append(f"\t{e_name} -- CUI:{ent_id.item()} ({ling_entity}) -- {node_name}")
                    # out_file.write(f"\t{e_name} -- CUI:{ent_id.item()} -- {node_name}\n")

            # #########################################################
            # # Validate concept_input_ids + concept_graph_edge_index #
            # #########################################################
            if concept_graph_edge_index is not None:
                concept_graph_src_ids = concept_graph_edge_index[0]
                concept_graph_trg_ids = concept_graph_edge_index[1]
                assert len(concept_graph_src_ids) == len(concept_graph_trg_ids) == len(rel_idx)
                check_strs_list.append("Graph edges:")
                for src_id, rel_id, trg_id in zip(concept_graph_src_ids, rel_idx, concept_graph_trg_ids):
                    ling_entity = random.choice(node_id2tokenized_terms[entity_node_ids[trg_id.item()]])
                    ling_entity = tokenizer.convert_ids_to_tokens(ling_entity, skip_special_tokens=True)
                    ling_entity = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in ling_entity))
                    s = (f"\t{node_names[src_id]} -- {id2rel[rel_id.item()]} --> {node_names[trg_id]} "
                         f"(CU?I: {entity_node_ids[trg_id].item()} ({ling_entity}))")
                    check_strs_list.append(s)
            new_line = '\n'
            out_file.write(f"{new_line.join(check_strs_list)}\n")
            out_file.write(">\n>\n>\n")
            # out_file.write("Linearized graph:\n")

            # # #######################################
            # # # Mention - concept name relation idx #
            # # #######################################
            # concept_graph_src_edge_index = concept_graph_edge_index[0]
            # concept_graph_trg_edge_index = concept_graph_edge_index[1]
            # # TODO: Do something here
            # head_node_names = [node_names[n] for n in concept_graph_src_edge_index]
            # tail_node_names = [node_names[n] for n in concept_graph_trg_edge_index]
            # assert len(head_node_names) == len(tail_node_names) == len(rel_idx)
            # check_strs_list.append("Graph edgesedges:")
            # new_line = '\n'
            # out_file.write(f"{new_line.join(check_strs_list)}\n")
            # TODO: Не надо ли ещё сделать assert на максимальный и минимальный номер вершины: все вершины из x
            # TODO: Должны встретиться в edge_index


def main(args):
    transformer_tokenizer = AutoTokenizer.from_pretrained(args.transformer_tokenizer_name)

    node_id2terms_path = args.node_id2terms_path
    graph_data_dir = args.graph_data_dir
    node_id2tokenized_terms = load_tokenized_concepts(node_id2terms_path)
    token_entity_index_type = args.token_entity_index_type
    tokenized_data_dir = args.tokenized_data_dir
    graph_format = args.graph_format
    use_rel_or_rela = args.use_rel_or_rela
    rela2rela_name_path = args.rela2rela_name
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    create_dir_if_not_exists(output_dir)
    if use_rel_or_rela == "rel":
        rel2id_path = os.path.join(graph_data_dir, "rel2id")
        rel2id = load_dict(rel2id_path, dtype_1=str, dtype_2=int)
    elif use_rel_or_rela == "rela":
        rela2id_path = os.path.join(graph_data_dir, "rela2id")
        rel2id = load_dict(rela2id_path, dtype_1=str, dtype_2=int)
    else:
        raise ValueError(f"Invalid use_rel_or_rela : {use_rel_or_rela}")
    id2rel = {v: k for k, v in rel2id.items()}

    MAX_N_NEIGHBORS = 3
    USE_REL = True
    SENTENCE_MAX_LENGTH = 128
    CONCEPT_MAX_LENGTH = 32
    MLM_PROBABILITY = 0.0
    MENTION_MASKING_PROB = 0.0
    CONCEPT_NAME_MASKING_PROB = 0.0
    lin_graph_max_length = 256
    BATCH_SIZE = args.batch_size

    output_fname = os.path.basename(output_path)
    output_dir = os.path.dirname(output_path)
    output_train_fname = f"train_{output_fname}"
    output_val_fname = f"val_{output_fname}"
    output_train_path = os.path.join(output_dir, output_train_fname)
    output_val_path = os.path.join(output_dir, output_val_fname)

    rela2id_path = os.path.join(graph_data_dir, "rela2id")
    rela2id = load_dict(rela2id_path, dtype_1=str, dtype_2=int)

    rel_id2tokenized_name = None
    if rela2rela_name_path is not None:
        rel2rel_name = load_dict(rela2rela_name_path, dtype_1=str, dtype_2=str)
        print("rela2id", rela2id)
        # rel_id2name = {rela2id[k]: v for k, v in rel2rel_name.items()}
        rel_id2name = {v: rel2rel_name[k] for k, v in rela2id.items()}
        rel_id2tokenized_name = {k: tuple(transformer_tokenizer.encode_plus(v,
                                                                            max_length=16,
                                                                            add_special_tokens=False,
                                                                            truncation=True)["input_ids"])
                                 for k, v in rel_id2name.items()}

    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")
    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, USE_REL, drop_selfloops=True,
                                                  use_rel_or_rela=use_rel_or_rela)
    logging.info(f"Processing training set...")
    tr_offsets, tr_offset_lowerbounds, tr_offset_upperbounds, tr_offset_filenames = load_offset_index(
        tokenized_data_dir,
        prefix="train")
    # TODO
    """
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
    """
    tr_dataset = TextGraphGraphNeighborsOffsetDataset(tokenizer=transformer_tokenizer,
                                                      input_data_dir=tokenized_data_dir,
                                                      offsets=tr_offsets,
                                                      offset_lowerbounds=tr_offset_lowerbounds,
                                                      offset_upperbounds=tr_offset_upperbounds,
                                                      offset_filenames=tr_offset_filenames,
                                                      node_id2adjacency_list=node_id2adjacency_list,
                                                      node_id2input_ids=node_id2tokenized_terms,
                                                      max_n_neighbors=MAX_N_NEIGHBORS,
                                                      token_entity_index_type=token_entity_index_type,
                                                      use_rel=USE_REL,
                                                      graph_format=graph_format,
                                                      rel_id2tokenized_name=rel_id2tokenized_name,
                                                      lin_graph_max_length=lin_graph_max_length,
                                                      sentence_max_length=SENTENCE_MAX_LENGTH,
                                                      concept_max_length=CONCEPT_MAX_LENGTH,
                                                      mlm_probability=MLM_PROBABILITY,
                                                      concept_name_masking_prob=CONCEPT_NAME_MASKING_PROB,
                                                      mention_masking_prob=MENTION_MASKING_PROB)
    tr_data_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, num_workers=0,
                                shuffle=True, collate_fn=tr_dataset.collate_fn)
    offset_data_sanity_check(data_loader=tr_data_loader,
                             tokenizer=transformer_tokenizer,
                             id2rel=id2rel,
                             node_id2tokenized_terms=node_id2tokenized_terms,
                             log_file_path=output_train_path)
    ###############################################################################
    ###############################################################################
    logging.info(f"Processing validation set...")
    val_offsets, val_offset_lowerbounds, val_offset_upperbounds, val_offset_filenames = load_offset_index(
        tokenized_data_dir,
        prefix="val")
    val_dataset = TextGraphGraphNeighborsOffsetDataset(tokenizer=transformer_tokenizer,
                                                       input_data_dir=tokenized_data_dir,
                                                       offsets=val_offsets,
                                                       offset_lowerbounds=val_offset_lowerbounds,
                                                       offset_upperbounds=val_offset_upperbounds,
                                                       offset_filenames=val_offset_filenames,
                                                       node_id2adjacency_list=node_id2adjacency_list,
                                                       node_id2input_ids=node_id2tokenized_terms,
                                                       max_n_neighbors=MAX_N_NEIGHBORS,
                                                       token_entity_index_type=token_entity_index_type,
                                                       use_rel=USE_REL,
                                                       graph_format=graph_format,
                                                       lin_graph_max_length=lin_graph_max_length,
                                                       rel_id2tokenized_name=rel_id2tokenized_name,
                                                       sentence_max_length=SENTENCE_MAX_LENGTH,
                                                       concept_max_length=CONCEPT_MAX_LENGTH,
                                                       mlm_probability=MLM_PROBABILITY,
                                                       concept_name_masking_prob=CONCEPT_NAME_MASKING_PROB,
                                                       mention_masking_prob=MENTION_MASKING_PROB)

    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0,
                                 shuffle=True, collate_fn=val_dataset.collate_fn)

    offset_data_sanity_check(data_loader=val_data_loader,
                             tokenizer=transformer_tokenizer,
                             id2rel=id2rel,
                             node_id2tokenized_terms=node_id2tokenized_terms,
                             log_file_path=output_val_path)
    # TODO: Проверить пограничные значения +/- 1

    # # TODO: КАК CUI-LESS ОКАЗАЛСЯ ВО ВХОДЕ?


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )

    parser = ArgumentParser()
    # parser.add_argument('--tokenized_data_dir',
    #                     default="/home/c204/University/NLP/BERN2_sample/TOKENIZED_UNMASKED")
    # parser.add_argument('--node_id2cui_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/id2cui")
    # parser.add_argument('--transformer_tokenizer_name', default="prajjwal1/bert-tiny")
    # parser.add_argument('--node_id2terms_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/node_id2terms_list")

    parser.add_argument('--tokenized_data_dir',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/tokenized_sentences/")
    parser.add_argument('--graph_data_dir',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph/")
    parser.add_argument('--transformer_tokenizer_name',
                        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--batch_size', default=32, type=int, required=False)
    parser.add_argument("--token_entity_index_type", type=str, choices=("edge_index", "matrix"),
                        required=False, default="edge_index")
    parser.add_argument("--graph_format", type=str, choices=("edge_index", "linear"),
                        required=False, default="edge_index")
    parser.add_argument("--use_rel_or_rela", type=str, required=False, choices=("rel", "rela"),
                        default="rel")
    parser.add_argument("--rela2rela_name", type=str, required=False)
    # parser.add_argument("--rela2rela_name", type=str, required=False,
    #                     default="/home/c204/University/NLP/text_kb/rela2rela_name_2020ab.tsv")
    # default="prajjwal1/bert-tiny")
    parser.add_argument('--node_id2terms_path',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--output_path',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/sanity_check_offset_dataset_edge_index_graph_token2entity.txt")

    args = parser.parse_args()
    main(args)
