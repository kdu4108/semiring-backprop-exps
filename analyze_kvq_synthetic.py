# Script to evaluate the top gradient paths for synthetic datasets.

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import ast
import brunoflow as bf
from brunoflow.ad.utils import check_node_equals_tensor, check_node_allclose_tensor
from jax import numpy as jnp
import numpy as np
import transformers
import torch
from transformers import BertTokenizerFast, BfBertForMaskedLM, BfBertForSequenceClassification, BertConfig
from collections import Counter, OrderedDict
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from kvq_utils import (
    convert_sentence_to_tokens_and_target_idx,
    find_matching_nodes,
    preprocessgrad_per_parent_per_word_data_per_layer,
    rename_matmul_kvq_nodes,
    summarize_max_grad_kvq,
    plot_max_grad_against_layer_per_word,
)
import wandb
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--MODEL_NAME", type=str, default="Bert-hs16-numheads2-bs32-lr0.001-n42-dropoutprob0.2")
    parser.add_argument("-i", "--EXAMPLE_INDICES_RANGE", nargs="+", type=int, default=[0, 1000])
    parser.add_argument(
        "-d",
        "--DATA_DIR",
        type=str,
        default="data/FirstTokenRepeatedOnce/FirstTokenRepeatedOnce-num_points10000-vs20-seqlen10/1.0-0/",
    )
    parser.add_argument("-a", "--AGG_METHOD", type=str, default="oplus")

    return parser.parse_args()


def process_sentence(dataset_path: str, example_index: int):
    df = pd.read_csv(dataset_path, names=["index", "sentence", "label"], header=0)
    index, sentence, label = df.iloc[example_index]
    tokens = ast.literal_eval(sentence)

    print(f"Analyzing sentence {example_index}: '{sentence}'")

    input_ids = np.expand_dims(tokens, axis=0)
    jax_input_ids = bf.Node(jnp.array(input_ids, dtype=int), name="inputs")

    return tokens, jax_input_ids, label


def analyze_kvq(bf_model, optimizer, dataset_path, example_index, agg_method, plots_dir, make_plot=False):
    (tokens, jax_input_ids, label) = process_sentence(dataset_path, example_index)

    # Forward pass of BfBertForMaskedLM
    optimizer.zero_gradients()
    bf_model.train(False)
    logits = bf_model(input_ids=jax_input_ids).logits  # shape = (vs, seq_len)
    correct_word_logit = logits[0, label]
    bf_model.train(True)

    # Backward Pass
    correct_word_logit.backprop(values_to_compute=("max_grad",))

    # Preprocessing KVQ nodes
    input_to_bert_attn_nodes = find_matching_nodes(correct_word_logit, "input to bertattention")
    rename_matmul_kvq_nodes(input_to_bert_attn_nodes)

    print(f"num layers in bert attn: {len(input_to_bert_attn_nodes)}")
    grad_per_parent_per_word_per_layer_df = preprocessgrad_per_parent_per_word_data_per_layer(
        input_to_bert_attn_nodes, tokens, agg_method=agg_method
    )

    if make_plot:
        save_path = os.path.join(plots_dir, f"{example_index}_max_grad_against_layer_per_word.png")
        plot_max_grad_against_layer_per_word(
            grad_per_parent_per_word_per_layer_df,
            tokens,
            save_path=save_path,
        )
        # After loading/preprocessing your dataset, log it as an artifact to W&B
        print(f"Logging plots to w&b run {wandb.run}.")
        wandb.log({"plot": wandb.Image(save_path)})

    # Augment the DF with extra info
    # get correct_word_logit, incorrect_word_logit
    correct_word_logit = logits[:, label].val[0].__array__()  # single np float
    incorrect_word_logit = logits[:, int(not label)].val[0].__array__()  # single np float
    is_correct = correct_word_logit > incorrect_word_logit
    grad_per_parent_per_word_per_layer_df["sentence_index"] = example_index
    grad_per_parent_per_word_per_layer_df["is_correct"] = is_correct
    grad_per_parent_per_word_per_layer_df["correct_pred_logit"] = correct_word_logit
    grad_per_parent_per_word_per_layer_df["incorrect_pred_logit"] = incorrect_word_logit
    grad_per_parent_per_word_per_layer_df["sentence"] = str(tokens)

    grad_per_parent_per_word_per_layer_df = grad_per_parent_per_word_per_layer_df[
        [
            "sentence_index",
            "is_correct",
            "correct_pred_logit",
            "incorrect_pred_logit",
            "token",
            "token_index",
            "layer_num",
            "path_name",
            "path grad",
            "sentence",
        ]
    ]
    return grad_per_parent_per_word_per_layer_df


def main():
    args = get_args()

    _, PROJECT_NAME = "kdu", "bauer-kvq-synthetic"  # set to your entity and project

    # PARAMETERS
    DATA_DIR = args.DATA_DIR
    MODEL_NAME = args.MODEL_NAME
    AGG_METHOD = args.AGG_METHOD
    MODEL_DIR = os.path.join(DATA_DIR, "models", args.MODEL_NAME)
    CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
    # MODEL_CONFIG_PATH = args.MODEL_CONFIG_PATH
    DATASET_PATH = os.path.join(DATA_DIR, "inputs", "val_data.csv")
    EXAMPLE_INDICES_RANGE = args.EXAMPLE_INDICES_RANGE

    params_to_log = vars(args)

    wandb.init(
        project=PROJECT_NAME,
        name=f"analyze_kvq_{EXAMPLE_INDICES_RANGE}_{datetime.now().isoformat(sep='_', timespec='seconds')}",
        config=params_to_log,
        tags=["analysis", AGG_METHOD],
    )
    print(dict(wandb.config))

    # Construct intermediate metadata variables
    results_dir = os.path.join(DATA_DIR, "results", "max_grad", AGG_METHOD, MODEL_NAME)
    plots_dir = os.path.join(results_dir, "plots")
    df_filename = (
        f"bert_synthetic_results_max_grad_{EXAMPLE_INDICES_RANGE[0]}_{EXAMPLE_INDICES_RANGE[1]}.csv"
        if EXAMPLE_INDICES_RANGE is not None
        else "bert_synthetic_results_max_grad.csv"
    )
    os.makedirs(plots_dir, exist_ok=True)

    # Load model
    config = BertConfig.from_pretrained(CONFIG_PATH)
    bf_model: BfBertForSequenceClassification = BfBertForSequenceClassification(config)
    bf_model.load_state_dict(torch.load(MODEL_PATH))

    optimizer = bf.opt.Adam(bf_model.parameters())
    output_path = os.path.join(results_dir, df_filename)
    print("Output path:", output_path)
    for i in range(EXAMPLE_INDICES_RANGE[0], EXAMPLE_INDICES_RANGE[1]):
        print(f"ANALYZING SENTENCE {i}\n\n")
        df = analyze_kvq(
            bf_model=bf_model,
            optimizer=optimizer,
            dataset_path=DATASET_PATH,
            example_index=i,
            agg_method=AGG_METHOD,
            plots_dir=plots_dir,
        )
        df.to_csv(output_path, mode="a", header=False)

    wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
