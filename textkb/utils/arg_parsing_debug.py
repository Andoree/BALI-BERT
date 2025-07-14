import argparse
from argparse import ArgumentParser

from textkb.utils.arg_parsing import add_gat_arguments


def parse_modular_biencoder_alignment_model_args_debug():
    parser = argparse.ArgumentParser()

    "/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/tokenized_sentences"
    parser.add_argument("--graph_data_dir", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph")
    parser.add_argument("--train_data_dir", type=str, required=False,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/tokenized_sentences")
    parser.add_argument("--validate", type=bool, default=True)
    parser.add_argument("--val_data_dir", type=str, required=False)
    # parser.add_argument("--val_data_dir", type=str, required=False,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/tokenized_sentences")
    parser.add_argument("--tokenized_concepts_path", type=str, required=False,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--output_dir", type=str, default="DELETE/")
    parser.add_argument("--train_sample_size", type=int, required=False)
    #
    parser.add_argument("--mlm_task", default=True, required=False)
    parser.add_argument("--textual_contrastive_task", default=False, required=False)
    parser.add_argument("--text_node_contrastive_task", default=False, required=False)
    parser.add_argument("--mention_concept_name_link_prediction_task", default=False, required=False)
    parser.add_argument("--text_graph_contrastive_task_central", action="store_true")
    parser.add_argument("--text_graph_contrastive_task_corrupted", action="store_true")
    parser.add_argument("--dgi_task", action="store_true")
    parser.add_argument("--text_graph_contrastive_task_mse", required=False,
                        default=True)
    parser.add_argument("--graph_mlm_task", required=False, default=False)
    parser.add_argument("--text_node_class_task", required=False, default=True)

    parser.add_argument("--cls_constraint_task", action="store_true")
    parser.add_argument("--concept_encoder_nograd", action="store_true")
    parser.add_argument("--bool_filter_token2entity", default=True)
    parser.add_argument("--graph_encoder_name", type=str, required=False,
                        choices=("gat", "graphsage"), default="gat")

    #
    parser.add_argument("--concept_name_masking_prob", type=float, default=0.15,
                        required=False)
    parser.add_argument("--mention_masking_prob", type=float, default=0.15, required=False)
    parser.add_argument("--mlm_probability", type=float, default=0.15, required=False)
    parser.add_argument("--class_neg_sampling_ratio", type=float, default=1.)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--sentence_emb_pooling", type=str, choices=("cls", "mean"), default="mean", required=False)
    parser.add_argument("--concept_emb_pooling", type=str, choices=("cls", "mean", "mean1"),
                        default="mean1", required=False)
    parser.add_argument("--link_transform_type", type=str, choices=("distmult", "transe", "rotate"),
                        default="distmult", required=False)
    parser.add_argument("--entity_aggregation_conv", type=str, choices=("mean", "gat", "attention"),
                        required=False, default="attention")
    parser.add_argument("--token_entity_index_type", type=str, choices=("edge_index", "matrix"),
                        required=False, default="matrix")
    parser.add_argument("--graph_format", type=str, choices=("edge_index", "linear"),
                        required=False, default="linear")
    parser.add_argument("--linear_graph_format", type=str, choices=("v1", "v2"),
                        required=False, default="v1")
    parser.add_argument("--remove_gat_output_dropout", action="store_true")

    parser.add_argument("--bert_encoder_name", default="prajjwal1/bert-tiny")
    # parser.add_argument("--bert_encoder_name",
    #                     default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--bert_learning_rate", type=float, default=1e-5)
    parser.add_argument("--graph_learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_num_warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_n_neighbors", type=int, default=1)
    parser.add_argument('--use_miner', action="store_true")
    parser.add_argument('--miner_margin', default=0.2, type=float)
    parser.add_argument("--sentence_max_length", type=int, default=128)
    parser.add_argument("--concept_max_length", type=int, default=32)
    parser.add_argument("--lin_graph_max_length", type=int, required=False, default=128)
    parser.add_argument("--freeze_embs", action="store_true")
    parser.add_argument("--freeze_layers", required=False, type=int)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--contrastive_loss", default="ms-loss",
                        choices=("ms-loss", "nceloss",), type=str)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0)
    parser.add_argument("--use_cuda", action="store_true", )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deepspeed", default=False)
    parser.add_argument("--deepspeed_cfg_path", required=False)
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--save_every_N_epoch", type=int, default=1)
    parser.add_argument("--save_every_N_steps", type=int, required=False, default=100000)
    parser.add_argument("--eval_every_N_steps", type=int, required=False, default=100)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument("--sentence_encoder_pooling_layer", type=int, required=False, default=-2)
    parser.add_argument("--concept_encoder_pooling_layer", type=int, required=False, default=-2)

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--use_rel", type=bool, default=True)
    parser.add_argument("--use_rel_or_rela", default="rela", choices=("rel", "rela"), type=str)
    parser.add_argument("--rela2rela_name", type=str, required=False,
                        default="/home/c204/University/NLP/text_kb/rela2rela_name_2020ab.tsv")
    parser.add_argument("--drop_loops", default=True)

    parser.add_argument("--ignore_tokenization_assert", default=True)
    parser.add_argument('--random_seed', type=int, default=29)
    parser.add_argument("--alive_prints", action="store_true")
    parser.add_argument("--intermodal_alignment_network", action="store_true")
    parser.add_argument('--output_debug_path', type=str, required=False, default="./DEBUG_IN_MODEL/debug_in_model.txt")

    add_gat_arguments_debug(parser)
    add_graphsage_arguments_debug(parser)

    args = parser.parse_args()
    return args


def add_gat_arguments_debug(parser: ArgumentParser):
    parser.add_argument("--gat_num_layers", type=int, default=2)
    parser.add_argument("--gat_num_hidden_channels", type=int, default=256)
    parser.add_argument("--gat_dropout_p", type=float, default=0.1)
    parser.add_argument("--gat_num_att_heads", type=int, default=2)
    parser.add_argument("--gat_attention_dropout_p", type=float, default=0.1)

    parser.add_argument("--gat_add_self_loops", default=True)


def add_graphsage_arguments_debug(parser: ArgumentParser):
    # parser.add_argument("--gat_num_layers", type=int, required=False)
    # parser.add_argument("--gat_num_hidden_channels", type=int, required=False)
    # parser.add_argument("--gat_dropout_p", type=float, required=False)
    parser.add_argument("--graphsage_project", action="store_true")
    parser.add_argument("--graphsage_normalize", action="store_true")
def parse_alignment_model_args_debug(graph_encoder_name):
    assert graph_encoder_name in ("GAT",)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--graph_data_dir", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph")
    parser.add_argument("--graph_data_dir", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug")
    parser.add_argument("--output_dir", type=str, default="DELETE/")

    # parser.add_argument("--bert_encoder_name", default="prajjwal1/bert-tiny")
    parser.add_argument("--bert_encoder_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--graph_model_type", choices=("shared_encoder", "gebert"),
                        default="gebert")
    parser.add_argument("--gebert_checkpoint_path", required=False,
                        default="/home/c204/University/NLP/BERN2_sample/gebert_checkpoint_for_debug/checkpoint_e_1_steps_94765.pth")

    # parser.add_argument("--tokenized_sentences_path", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/TOKENIZED_UNMASKED")
    # parser.add_argument("--tokenized_concepts_path", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/node_id2terms_list_tinybert_test")
    parser.add_argument("--train_tokenized_sentences_path", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences")  # ")
    parser.add_argument("--val_tokenized_sentences_path", type=str, required=False,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences")
    parser.add_argument("--tokenized_concepts_path", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--bert_learning_rate", type=float, default=1e-5)
    parser.add_argument("--graph_learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_num_warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_n_neighbors", type=int, default=1)
    parser.add_argument("--sentence_max_length", type=int, default=128)
    parser.add_argument("--use_rel", action="store_true", )
    parser.add_argument('--use_miner', action="store_true")
    parser.add_argument('--miner_margin', default=0.2, type=float)
    parser.add_argument("--freeze_graph_bert_encoder", default=True)
    parser.add_argument("--freeze_graph_encoder", default=True)
    # TODO remove_selfloops
    parser.add_argument("--remove_selfloops", action="store_true")
    parser.add_argument("--concept_max_length", type=int, default=32)
    parser.add_argument("--masking_mode", choices=("text", "graph", "both", "random"), type=str,
                        default="random")
    parser.add_argument("--contrastive_loss", default="ms-loss",
                        choices=("ms-loss", "nceloss",), type=str)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0)
    parser.add_argument("--use_cuda", action="store_true", )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deepspeed", default=True)
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--save_every_N_epoch", type=int, default=1)
    parser.add_argument("--save_every_N_steps", type=int, required=False, default=100000)
    parser.add_argument("--eval_every_N_steps", type=int, required=False, default=500)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    if graph_encoder_name == "GAT":
        add_gat_arguments_debug(parser)

    args = parser.parse_args()
    return args


def parse_modular_alignment_model_args_debug():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data_dir", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug")
    parser.add_argument("--train_data_dir", type=str,
                        default="/home/c204/University/NLP/text_kb/textkb/textkb/preprocessing/DELETE/train")
    parser.add_argument("--val_data_dir", type=str, required=False,
                        default="/home/c204/University/NLP/text_kb/textkb/textkb/preprocessing/DELETE/train")
    parser.add_argument("--node_embs_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default="DELETE/")

    parser.add_argument("--mlm_task", default=True)
    parser.add_argument("--contrastive_task", default=True)
    parser.add_argument("--graph_lp_task", default=True)
    parser.add_argument("--intermodal_lp_task", default=True)
    parser.add_argument("--freeze_node_embs", default=True)

    parser.add_argument("--entity_masking_prob", type=float, default=0.15)  # TODO: Это ещё обдумать
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--link_score_type", type=str, choices=("transe", "distmult"),
                        default="transe")
    parser.add_argument("--link_negative_sample_size", type=int, default=2)
    parser.add_argument("--link_regularizer_weight", type=float, default=0.01)
    parser.add_argument("--embedding_transform", type=str, default="fc", choices=("static", "fc"))

    parser.add_argument("--bert_encoder_name", default="prajjwal1/bert-tiny")
    # default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--bert_learning_rate", type=float, default=1e-5)
    parser.add_argument("--graph_learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_num_warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--sentence_max_length", type=int, default=128)
    parser.add_argument("--contrastive_loss", default="ms-loss",
                        choices=("ms-loss", "infonce",), type=str)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0)
    parser.add_argument("--use_cuda", action="store_true", )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--save_every_N_epoch", type=int, default=1)
    parser.add_argument("--save_every_N_steps", type=int, required=False, default=100000)
    parser.add_argument("--eval_every_N_steps", type=int, required=False, default=500)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()
    return args
