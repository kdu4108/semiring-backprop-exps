# Script to evaluate the top gradient paths for LGD's subject-verb agreement dataset.

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import brunoflow as bf
from brunoflow.ad.utils import check_node_equals_tensor, check_node_allclose_tensor
from jax import numpy as jnp
import numpy as np
import transformers
import torch
from transformers import BertTokenizerFast, BfBertForMaskedLM
from collections import Counter, OrderedDict
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
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
    parser.add_argument("-m", "--MODEL_ID", type=str, default="google/bert_uncased_L-6_H-128_A-2")
    parser.add_argument("-i", "--EXAMPLE_INDICES_RANGE", nargs="+", type=int, default=[0, 10])
    parser.add_argument("-d", "--DATASET_PATH", type=str, default="data/lgd_sva/lgd_max_grad_eval_dataset.tsv")
    parser.add_argument("-a", "--AGG_METHOD", type=str, default="oplus")

    return parser.parse_args()


def process_sentence(dataset_path: str, example_index: int, tokenizer):
    df = pd.read_csv(dataset_path, sep="\t", names=["type", "sentence", "masked_sentence", "good_word", "bad_word"])
    num_distractors, _, masked_sentence, good, bad = df.iloc[example_index]
    print(f"Analyzing sentence {example_index}: '{masked_sentence}'")

    # Tokenize text and pass into model
    word_ids = tokenizer.convert_tokens_to_ids([good, bad])
    tokens, target_idx = convert_sentence_to_tokens_and_target_idx(masked_sentence, tokenizer)
    input_ids = np.expand_dims(tokenizer.convert_tokens_to_ids(tokens), axis=0)
    jax_input_ids = bf.Node(jnp.array(input_ids, dtype=int), name="inputs")

    return word_ids, tokens, target_idx, input_ids, jax_input_ids, masked_sentence, good, bad, num_distractors


def analyze_kvq(
    bf_model, tokenizer, dataset_path: str, example_index: int, agg_method: str, plots_dir: str, make_plot=False
):
    (
        word_ids,
        tokens,
        target_idx,
        input_ids,
        jax_input_ids,
        masked_sentence,
        good,
        bad,
        num_distractors,
    ) = process_sentence(dataset_path, example_index, tokenizer)

    # Forward pass of BfBertForMaskedLM
    bf_model.train(False)
    out_bf = bf_model(input_ids=jax_input_ids).logits  # shape = (vs, seq_len)
    qoi = out_bf[:, target_idx]  # shape = (1, vs)
    qoi = qoi[:, word_ids[0]] - qoi[:, word_ids[1]]  # shape = (1,)
    bf_model.train(True)

    # Backward Pass
    qoi.backprop(values_to_compute=("max_grad",))

    # Preprocessing KVQ nodes
    input_to_bert_attn_nodes = find_matching_nodes(out_bf[0], "input to bertattention")
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
    correct_word_logit = out_bf[:, target_idx][:, word_ids[0]].val[0].__array__()  # single np float
    incorrect_word_logit = out_bf[:, target_idx][:, word_ids[1]].val[0].__array__()  # single np float
    is_correct = correct_word_logit > incorrect_word_logit
    grad_per_parent_per_word_per_layer_df["sentence_index"] = example_index
    grad_per_parent_per_word_per_layer_df["is_correct"] = is_correct
    grad_per_parent_per_word_per_layer_df["num_distractors"] = num_distractors
    grad_per_parent_per_word_per_layer_df["correct_word"] = good
    grad_per_parent_per_word_per_layer_df["correct_word_logit"] = correct_word_logit
    grad_per_parent_per_word_per_layer_df["incorrect_word"] = bad
    grad_per_parent_per_word_per_layer_df["incorrect_word_logit"] = incorrect_word_logit
    grad_per_parent_per_word_per_layer_df["masked_sentence"] = masked_sentence

    grad_per_parent_per_word_per_layer_df = grad_per_parent_per_word_per_layer_df[
        [
            "sentence_index",
            "is_correct",
            "num_distractors",
            "correct_word",
            "correct_word_logit",
            "incorrect_word",
            "incorrect_word_logit",
            "token",
            "token_index",
            "layer_num",
            "path_name",
            "path grad",
            "masked_sentence",
        ]
    ]
    return grad_per_parent_per_word_per_layer_df


def main():
    args = get_args()

    _, PROJECT_NAME = "kdu", "bauer-kvq"  # set to your entity and project

    # PARAMETERS
    MODEL_ID = args.MODEL_ID
    AGG_METHOD = args.AGG_METHOD
    EXAMPLE_INDICES_RANGE = args.EXAMPLE_INDICES_RANGE
    DATASET_PATH = args.DATASET_PATH
    DATA_DIR = "data/lgd_sva"

    params_to_log = vars(args)

    wandb.init(
        project=PROJECT_NAME,
        name=f"analyze_kvq_{EXAMPLE_INDICES_RANGE}_{datetime.now().isoformat(sep='_', timespec='seconds')}",
        config=params_to_log,
        tags=["analysis", AGG_METHOD],
    )
    print(dict(wandb.config))

    # Construct intermediate metadata variables
    results_dir = os.path.join(DATA_DIR, "bf-" + MODEL_ID, "results", "max_grad", AGG_METHOD)
    plots_dir = os.path.join(results_dir, "plots")
    df_filename = (
        f"lgd_results_base_max_grad_{EXAMPLE_INDICES_RANGE[0]}_{EXAMPLE_INDICES_RANGE[1]}.csv"
        if EXAMPLE_INDICES_RANGE is not None
        else "lgd_results_max_grad_base.csv"
    )
    os.makedirs(plots_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(MODEL_ID)
    bf_model: BfBertForMaskedLM = BfBertForMaskedLM.from_pretrained(MODEL_ID)

    output_path = os.path.join(results_dir, df_filename)
    print("Output path:", output_path)

    for i in range(EXAMPLE_INDICES_RANGE[0], EXAMPLE_INDICES_RANGE[1]):
        print(f"ANALYZING SENTENCE {i}\n\n")
        df = analyze_kvq(
            bf_model=bf_model,
            tokenizer=tokenizer,
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
