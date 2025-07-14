import logging
import os.path

import numpy as np
import argparse

import torch
from textkb.utils.io import create_dir_if_not_exists


def main(args):
    input_checkpoint_dir = args.input_checkpoint_dir
    output_checkpoint_dir = args.output_checkpoint_dir
    create_dir_if_not_exists(output_checkpoint_dir)

    train_state_path = os.path.join(input_checkpoint_dir, "train_state.pth")

    checkpoint = torch.load(train_state_path, map_location=lambda storage, loc: storage)
    new_checkpoint = {
        "epoch": checkpoint["epoch"],
        "num_steps": checkpoint["num_steps"],
        "bert_encoder": checkpoint["bert_encoder"],
        "optimizer": checkpoint["optimizer"].state_dict(),
        "scheduler": checkpoint["scheduler"].state_dict(),
        "scaler": checkpoint["scaler"].state_dict()

    }
    new_checkpoint_path = os.path.join(output_checkpoint_dir, "train_state.pth")
    torch.save(new_checkpoint, new_checkpoint_path)

    input_node_embs_path = os.path.join(input_checkpoint_dir, "node_embs_matrix.pt")
    node_embs_matrix = torch.load(input_node_embs_path, map_location=lambda storage, loc: storage)
    output_node_embs_path = os.path.join(output_checkpoint_dir, "node_embs_matrix.pt")

    torch.save(node_embs_matrix.state_dict(), output_node_embs_path)

    # node_embs_matrix = torch.load(input_node_embs_path, map_location=lambda storage, loc: storage)
    # node_embs_matrix = node_embs_matrix.cpu().detach().numpy()
    #
    # output_node_embs_path = os.path.join(output_checkpoint_dir, "node_embs.npy")
    # np.save(output_node_embs_path, node_embs_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_checkpoint_dir', type=str, )
    parser.add_argument('--output_checkpoint_dir', type=str, )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
