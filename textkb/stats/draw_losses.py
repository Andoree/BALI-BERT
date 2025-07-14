import logging
import os
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

from textkb.utils.io import create_dir_if_not_exists


def main(args):
    input_pretrained_models_path = args.input_pretrained_models_path
    output_graphs_dir = args.output_graphs_dir
    create_dir_if_not_exists(output_graphs_dir)

    for setup_name in os.listdir(input_pretrained_models_path):
        input_setup_dir = os.path.join(input_pretrained_models_path, setup_name)
        output_setup_dir = os.path.join(output_graphs_dir, setup_name)
        if not os.path.isdir(input_setup_dir):
            continue
        create_dir_if_not_exists(output_setup_dir)

        for i in range(5):
            losses_fname = f"losses_epoch_{i}"
            input_losses_path = os.path.join(input_setup_dir, losses_fname)
            if not os.path.exists(input_losses_path):
                continue
            if not os.path.exists(os.path.join(output_graphs_dir, setup_name)):
                os.makedirs(os.path.join(output_graphs_dir, setup_name))
            print(input_losses_path)
            losses_df = pd.read_csv(input_losses_path, sep='\t', on_bad_lines='warn')
            for col_loss_name in losses_df.columns:
                q = losses_df[col_loss_name].quantile(0.99)
                losses_df[losses_df[col_loss_name] < q][col_loss_name].plot.line()

                # losses_df[col_loss_name].plot.line()
                output_plot_path = os.path.join(output_graphs_dir, setup_name,  f"{losses_fname}_{col_loss_name}")
                plt.savefig(output_plot_path, format="pdf")
                





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_pretrained_models_path')
    parser.add_argument('--output_graphs_dir')

    args = parser.parse_args()
    main(args)
