import argparse
from argparse import ArgumentParser


def parse_modular_biencoder_alignment_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data_dir", type=str)
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val_data_dir", type=str, required=False)
    parser.add_argument("--tokenized_concepts_path", type=str)
    # parser.add_argument("--node_embs_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_sample_size", type=int, required=False)

    parser.add_argument("--mlm_task", action="store_true")
    parser.add_argument("--textual_contrastive_task", action="store_true")
    parser.add_argument("--text_node_contrastive_task", action="store_true")
    parser.add_argument("--text_graph_contrastive_task_corrupted", action="store_true")
    parser.add_argument("--text_graph_contrastive_task_mse", action="store_true")
    parser.add_argument("--graph_mlm_task", action="store_true")
    parser.add_argument("--cls_constraint_task", action="store_true")
    parser.add_argument("--text_node_class_task", action="store_true")

    parser.add_argument("--mention_concept_name_link_prediction_task", action="store_true")
    parser.add_argument("--text_graph_contrastive_task_central", action="store_true")
    parser.add_argument("--dgi_task", action="store_true")
    parser.add_argument("--graph_encoder_name", type=str, required=False,
                        choices=("gat", "graphsage"), default="gat")


    # parser.add_argument("--contrastive_task", action="store_true")
    # parser.add_argument("--graph_lp_task", action="store_true")
    # parser.add_argument("--intermodal_lp_task", action="store_true")
    # parser.add_argument("--freeze_node_embs", action="store_true")
    # parser.add_argument("--embedding_transform", type=str, choices=("static", "fc"))

    parser.add_argument("--concept_name_masking_prob", type=float, default=0.15)
    parser.add_argument("--mention_masking_prob", type=float, default=0.15)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--class_neg_sampling_ratio", type=float, default=1.)

    parser.add_argument("--sentence_emb_pooling", type=str, choices=("cls", "mean"))
    parser.add_argument("--concept_emb_pooling", type=str, choices=("cls", "mean", "mean1"))
    parser.add_argument("--link_transform_type", type=str, choices=("distmult", "transe", "rotate"),
                        required=False)
    parser.add_argument("--entity_aggregation_conv", type=str, choices=("mean", "gat", "attention",
                                                                        "weighted"), required=False, default="mean")
    parser.add_argument("--token_entity_index_type", type=str, choices=("edge_index", "matrix"),
                        required=False, default="edge_index")
    parser.add_argument("--graph_format", type=str, choices=("edge_index", "linear"),
                        required=False, default="edge_index")
    parser.add_argument("--linear_graph_format", type=str, choices=("v1", "v2"),
                        required=False, default="v1")

    parser.add_argument("--remove_gat_output_dropout", action="store_true")
    parser.add_argument("--freeze_embs", action="store_true")
    parser.add_argument("--freeze_layers", required=False, type=int)
    parser.add_argument("--concept_encoder_nograd", action="store_true")
    parser.add_argument("--sentence_encoder_pooling_layer", type=int, required=False, default=-1)
    parser.add_argument("--concept_encoder_pooling_layer", type=int, required=False, default=-1)

    # parser.add_argument("--link_score_type", type=str, choices=("transe", "distmult"))
    # parser.add_argument("--link_negative_sample_size", type=int)
    # parser.add_argument("--link_regularizer_weight", type=float, default=0.01)

    parser.add_argument("--bert_encoder_name")

    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--bert_learning_rate", type=float)

    parser.add_argument("--graph_learning_rate", type=float)
    parser.add_argument("--max_num_warmup_steps", type=int)
    parser.add_argument("--warmup_steps_ratio", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--sentence_max_length", type=int)
    parser.add_argument("--concept_max_length", type=int)
    parser.add_argument("--lin_graph_max_length", type=int, required=False)
    parser.add_argument("--rela2rela_name", type=str, required=False)
    parser.add_argument("--drop_loops", action="store_true")
    parser.add_argument("--bool_filter_token2entity", action="store_true")

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--contrastive_loss",
                        choices=("ms-loss", "infonce",), type=str)
    parser.add_argument("--max_n_neighbors", type=int)
    parser.add_argument("--use_rel", action="store_true")
    parser.add_argument('--use_miner', action="store_true")
    parser.add_argument('--miner_margin', default=0.2, type=float)
    parser.add_argument("--use_rel_or_rela", default="rel", choices=("rel", "rela"), type=str)
    parser.add_argument("--contrastive_loss_weight", type=float)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_cfg_path", required=False)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--save_every_N_epoch", type=int, default=1)
    parser.add_argument("--eval_every_N_steps", type=int, required=False)
    parser.add_argument("--save_every_N_steps", type=int, required=False)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--random_seed', type=int, default=29)
    parser.add_argument("--ignore_tokenization_assert", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--alive_prints", action="store_true")
    parser.add_argument("--intermodal_alignment_network", action="store_true")
    parser.add_argument('--output_debug_path', type=str, required=False)

    add_gat_arguments(parser)
    add_graphsage_arguments(parser)

    args = parser.parse_args()
    return args


def add_gat_arguments(parser: ArgumentParser):
    parser.add_argument("--gat_num_layers", type=int, required=False)
    parser.add_argument("--gat_num_hidden_channels", type=int, required=False)
    parser.add_argument("--gat_dropout_p", type=float, required=False)
    parser.add_argument("--gat_num_att_heads", type=int, required=False)
    parser.add_argument("--gat_attention_dropout_p", type=float, required=False)
    parser.add_argument("--gat_add_self_loops", action="store_true")


def add_graphsage_arguments(parser: ArgumentParser):
    # parser.add_argument("--gat_num_layers", type=int, required=False)
    # parser.add_argument("--gat_num_hidden_channels", type=int, required=False)
    # parser.add_argument("--gat_dropout_p", type=float, required=False)
    parser.add_argument("--graphsage_project", action="store_true")
    parser.add_argument("--graphsage_normalize", action="store_true")

def parse_alignment_model_args(graph_encoder_name):
    assert graph_encoder_name in ("GAT",)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--bert_encoder_name")
    parser.add_argument("--graph_model_type", choices=("shared_encoder", "gebert"), required=True)
    parser.add_argument("--gebert_checkpoint_path", required=False)

    parser.add_argument("--train_tokenized_sentences_path", type=str)
    parser.add_argument("--val_tokenized_sentences_path", type=str, default=None, required=False)
    parser.add_argument("--tokenized_concepts_path", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--bert_learning_rate", type=float)
    parser.add_argument("--graph_learning_rate", type=float)
    parser.add_argument("--max_num_warmup_steps", type=int)
    parser.add_argument("--warmup_steps_ratio", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_n_neighbors", type=int)
    parser.add_argument("--sentence_max_length", type=int)
    parser.add_argument("--use_rel", action="store_true")
    parser.add_argument("--freeze_graph_bert_encoder", action="store_true")
    parser.add_argument("--freeze_graph_encoder", action="store_true")
    parser.add_argument("--remove_selfloops", action="store_true")
    parser.add_argument("--concept_max_length", type=int)
    parser.add_argument("--masking_mode", choices=("text", "graph", "both", "random"), type=str)
    parser.add_argument("--contrastive_loss",
                        choices=("ms-loss", "nceloss",), type=str)
    parser.add_argument("--contrastive_loss_weight", type=float)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")

    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--save_every_N_epoch", type=int, default=1)
    parser.add_argument("--eval_every_N_steps", type=int, required=False)
    parser.add_argument("--save_every_N_steps", type=int, required=False)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument("--batch_size", type=int, )

    if graph_encoder_name == "GAT":
        add_gat_arguments(parser)

    args = parser.parse_args()
    return args


def parse_modular_alignment_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data_dir", type=str)
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--val_data_dir", type=str, required=False)
    parser.add_argument("--node_embs_path", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--mlm_task", action="store_true")
    parser.add_argument("--contrastive_task", action="store_true")
    parser.add_argument("--graph_lp_task", action="store_true")
    parser.add_argument("--intermodal_lp_task", action="store_true")
    parser.add_argument("--freeze_node_embs", action="store_true")
    parser.add_argument("--embedding_transform", type=str, choices=("static", "fc"))

    parser.add_argument("--entity_masking_prob", type=float, default=0.15)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--link_score_type", type=str, choices=("transe", "distmult"))
    parser.add_argument("--link_negative_sample_size", type=int)
    parser.add_argument("--link_regularizer_weight", type=float, default=0.01)

    parser.add_argument("--bert_encoder_name")

    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--bert_learning_rate", type=float)

    parser.add_argument("--graph_learning_rate", type=float)
    parser.add_argument("--max_num_warmup_steps", type=int)
    parser.add_argument("--warmup_steps_ratio", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--sentence_max_length", type=int)
    parser.add_argument("--contrastive_loss",
                        choices=("ms-loss", "infonce",), type=str)
    parser.add_argument("--contrastive_loss_weight", type=float)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_cfg_path", required=False)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--save_every_N_epoch", type=int, default=1)
    parser.add_argument("--eval_every_N_steps", type=int, required=False)
    parser.add_argument("--save_every_N_steps", type=int, required=False)
    parser.add_argument('--model_checkpoint_path', required=False, default=None)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--random_seed', type=int, default=29)
    parser.add_argument("--ignore_tokenization_assert", action="store_true")
    parser.add_argument("--sentence_encoder_pooling_layer", type=int, required=False, default=-1)
    parser.add_argument("--concept_encoder_pooling_layer", type=int, required=False, default=-1)

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    args = parser.parse_args()
    return args
