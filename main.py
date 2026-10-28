import json
import logging
import os
import time
from argparse import ArgumentParser
import datetime

import config


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    parser.add_argument(
        "--dataset", default="indtrack1", help="indtrack1-7, test357,535,560"
    )
    parser.add_argument("--K", default=4, help="10,50,90")
    parser.add_argument(
        "--strategy",
        default="interactive",
        help="concat, solo, inter, two, interactive, set_transformer, gat",
    )
    parser.add_argument(
        "--conditional_topk",
        default=None,
        help="on/off switch for conditional Stage-I decoding",
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)
    if options.conditional_topk is not None:
        os.environ["MC2P_CONDITIONAL_TOPK"] = str(options.conditional_topk)

    # Import only the trainer needed by the selected dataset.
    if options.mode == "train":
        if options.dataset == 'SP500':
            raise ValueError("The optional SP500 training script is not included in the core repository.")
        elif options.dataset.startswith('test'):
            import train_SSOA

            train_SSOA.train_supplier_selection(options.dataset, options.K, options.strategy)
        elif options.dataset.startswith('redset'):
            import train_redset

            train_redset.train_redset(options.dataset, options.K, options.strategy)
        else:
            import train

            train.train_stock_trading(options.dataset, options.K, options.strategy)


if __name__ == "__main__":
    main()
