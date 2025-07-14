import logging
import os
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer, AutoModel

from textkb.utils.io import load_dict, save_dict, save_bert_encoder_tokenizer_config, create_dir_if_not_exists


def save_bert_encoder_from_graph_model(pretrained_graph_model_dir: str, bert_initialization_model: str,
                                       checkpoint, output_dir: str):
    model_description_path = os.path.join(pretrained_graph_model_dir, "model_description.tsv")
    # model_checkpoint_path = os.path.join(pretrained_graph_model_dir, checkpoint_path)
    model_parameters_dict = load_dict(path=model_description_path, )
    pretrained_model_encoder_path_or_name = os.path.basename(model_parameters_dict["text_encoder"])
    if pretrained_model_encoder_path_or_name not in bert_initialization_model:
        raise ValueError(f"Pretrained graph-based encoder does not match with initialized model: "
                         f"{pretrained_model_encoder_path_or_name} not in {bert_initialization_model}")

    tokenizer = AutoTokenizer.from_pretrained(bert_initialization_model)
    bert_encoder = AutoModel.from_pretrained(bert_initialization_model).cpu()

    bert_encoder.load_state_dict(checkpoint["model_state"])

    save_bert_encoder_tokenizer_config(bert_encoder=bert_encoder, bert_tokenizer=tokenizer, save_path=output_dir)
    save_dict(save_path=os.path.join(output_dir, "model_description.tsv"), dictionary=model_parameters_dict)


def save_graph_encoder_state_from_checkpoint(checkpoint, output_dir: str):
    graph_encoder_state = checkpoint["graph_encoder"]
    output_graph_state_path = os.path.join(output_dir, "graph_encoder_state.pt")

    torch.save(graph_encoder_state, output_graph_state_path)


def main(args):
    input_model_dir = args.input_model_dir
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)

    for fname in os.listdir(input_model_dir):
        if not fname.startswith('checkpoint'):
            continue
        checkpoint_path = os.path.join(input_model_dir, fname)
        # checkpoint = torch.load(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        checkpoint_name = fname.split('.')[0]

        output_checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        save_bert_encoder_from_graph_model(pretrained_graph_model_dir=input_model_dir,
                                           bert_initialization_model=args.bert_initialization_model,
                                           checkpoint=checkpoint,
                                           output_dir=output_checkpoint_dir)
        save_graph_encoder_state_from_checkpoint(checkpoint=checkpoint, output_dir=output_checkpoint_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--input_model_dir')
    arg_parser.add_argument('--bert_initialization_model')
    arg_parser.add_argument('--output_dir')

    args = arg_parser.parse_args()
    main(args)
