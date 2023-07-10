# Helper script containing utility functions to compute the top gradient paths of a model.

import brunoflow as bf
from collections import Counter, OrderedDict
from jax import numpy as jnp
from typing import List, Tuple, Optional, Dict
import pandas as pd
import math
from matplotlib import pyplot as plt
import seaborn as sns
import os

TokenIndex = int
NodeName = str


def convert_sentence_to_tokens_and_target_idx(sent: str, tokenizer):
    pre, target, post = sent.split("***")
    if "mask" in target.lower():
        target = ["[MASK]"]
    else:
        target = tokenizer.tokenize(target)
    tokens = ["[CLS]"] + tokenizer.tokenize(pre)
    target_idx = len(tokens)
    # print(target_idx)
    tokens += target + tokenizer.tokenize(post) + ["[SEP]"]
    return tokens, target_idx


def find_matching_nodes(root: bf.Node, name: str):
    def _find_matching_nodes(root: bf.Node, name: str, visited=set()):
        assert isinstance(root, bf.Node), f"root input must be a Node, instead received {root}"
        if root in visited:
            return []

        matching_nodes = []
        if root.name is not None and name in root.name:
            matching_nodes.append(root)
            # return [root]
        for inp in root.inputs:
            if isinstance(inp, bf.Node):
                matching_nodes_in_subtree = _find_matching_nodes(inp, name, visited=visited)
                visited.add(inp)
                if matching_nodes_in_subtree:
                    matching_nodes = matching_nodes_in_subtree + matching_nodes

        return matching_nodes

    return _find_matching_nodes(root, name, visited=set())


def rename_matmul_kvq_nodes(input_to_bert_attn_nodes: List[bf.Node]):
    # Distinguish the matmul parents
    for input_to_bert_attn_layer in input_to_bert_attn_nodes:
        for parent in input_to_bert_attn_layer.get_parents():
            curr_parent = parent
            if curr_parent.name == "matmul":
                while "bertselfattention" not in curr_parent.name:
                    assert len(curr_parent.get_parents()) == 1
                    curr_parent = curr_parent.get_parents()[0]
                parent.name = f"matmul ({curr_parent.name})"


def compute_oplus_max_grad(input_to_bert_attn_node: bf.Node, token_idx: int) -> Dict[NodeName, float]:
    """
    Compute the max grad per each parent of input_to_bert_attn_node for a given token_idx, using the oplus max method.

    Returns:
        Dict containing 4 items, where each entry maps from a branch name to its single max grad value across all embedding units.
        e.g. ,
        {
            'matmul (bertselfattention value)': 0.0,
            'combine self_attention_output and bert attention input 8733937785884': 0.0,
            'matmul (bertselfattention key)': DeviceArray(0.00114273, dtype=float32),
            'matmul (bertselfattention mixed_query_layer)': 0.0
        }
    """
    parent_names: List[NodeName] = [p.name for p in input_to_bert_attn_node.get_parents()]
    parent_to_max_grad_val = dict.fromkeys(parent_names, 0.0)

    max_grads_per_embedding_unit = input_to_bert_attn_node.max_grad_of_output_wrt_node[0][0][
        token_idx
    ]  # first 0 is the grad value, second 0 is the 0th element in batch, 3rd index is the token # shape: (hidden_sz,)
    max_grad_parents_per_embedding_unit = input_to_bert_attn_node.max_grad_of_output_wrt_node[1][0][
        token_idx
    ]  # first 1 is the actual parent node, second 0 is the 0th element in batch, 3rd index is the token # shape: (hidden_sz,)
    emb_unit_with_max_grad_idx: int = jnp.argmax(max_grads_per_embedding_unit)  # value between [0, hidden_sz - 1]
    max_grad_across_embedding: float = max_grads_per_embedding_unit[emb_unit_with_max_grad_idx]
    max_grad_parent_across_embedding = max_grad_parents_per_embedding_unit[emb_unit_with_max_grad_idx]

    parent_to_max_grad_val[max_grad_parent_across_embedding.name] = max_grad_across_embedding

    return parent_to_max_grad_val


def compute_bucket_sum_max_grad(input_to_bert_attn_node: bf.Node, token_idx: int) -> Dict[NodeName, float]:
    """
    Compute the max grad per each parent of input_to_bert_attn_node for a given token_idx, using the bucket sum method.

    Returns:
        Dict containing 4 items, where each entry maps from a branch name to its accumulated max grad value across the embedding dimension.
    """
    parent_names: List[NodeName] = [p.name for p in input_to_bert_attn_node.get_parents()]
    max_grad_parent_for_emb = input_to_bert_attn_node.get_max_grad_parent()[0, token_idx]  # shape = (emb_sz,)
    parent_to_max_grad_val = dict.fromkeys(parent_names, 0.0)
    for j in range(len(max_grad_parent_for_emb)):
        emb_unit_max_grad_val = input_to_bert_attn_node.max_grad_of_output_wrt_node[0][0][token_idx][j]
        emb_unit_max_grad_parent = input_to_bert_attn_node.max_grad_of_output_wrt_node[1][0][token_idx][j]
        parent_to_max_grad_val[emb_unit_max_grad_parent.name] += emb_unit_max_grad_val

    return parent_to_max_grad_val


def summarize_max_grad_kvq(input_to_bert_attn_node: bf.Node, agg_method: str = "oplus"):
    """
    Args:
        input_to_bert_attn_node - the node which branches into k/v/q/skip. Has shape (bs=1, seq_len, hidden_sz)
        agg_method - how to aggregate max grads across the embedding for a given token.
            When agg_method is `oplus`, aggregate by taking the single (max_grad, max_grad_parent) across ALL embedding units.
            This sets output[token][max_grad_parent.name] = max_grad and output[token][other_max_grad_parent.name] = 0 for all other_max_grad_parents.
            This is more technically consistent with the entropy method.
            When agg_method is `bucket_sum`, aggregate by summing the (max_grad, max_grad_parent) for each embedding unit.
            This sets output[token][max_grad_parent.name] += max_grad for each embedding unit's (max_grad, max_grad_parent).
            This is like computing the top gradient path of each embedding unit and then computing a summary statistic,
            which may be more informative than simply taking the max across all units.

    Return:
        An OrderedDict with seq_len number of entries.
        Each entry is a dict containing 4 keys,
        where each kv-pair maps from a parent node name/branch (str name, e.g. `matmul (bertselfattention value)`) to
        a float representing the value for that branch.
    """
    if agg_method == "oplus":
        compute_max_grad_for_token = compute_oplus_max_grad
    elif agg_method == "bucket_sum":
        compute_max_grad_for_token = compute_bucket_sum_max_grad
    else:
        raise ValueError(f"agg_method must be one of `oplus` or `bucket_sum`, instead received {agg_method}.")

    if input_to_bert_attn_node.shape[0] != 1:
        raise ValueError(
            "summarize_max_grad_kvq expects a batch size of 1 for computing the max grad across k/v/q/skip and will only compute for 1st index in batch."
        )

    # Number of hidden units corresponding to each max grad parent option (for the input to bert attention Node)
    count_hidden_unit_max_grad_parents: OrderedDict[TokenIndex, List[Tuple[NodeName, int]]] = OrderedDict(
        {
            i: [(k.name, v) for k, v in Counter(input_to_bert_attn_node.get_max_grad_parent()[0, i]).items()]
            for i in range(len(input_to_bert_attn_node.get_max_grad_parent()[0]))
        }
    )  # keys are tokens, values are a list of the counts of # emb units for that token which have each of the possible max grad parents

    # Compute the max grad per parent node for each token
    parent_to_max_grad_val_for_all_words = []
    for i in range(len(input_to_bert_attn_node.get_max_grad_parent()[0])):  # each word
        parent_to_max_grad_val = compute_max_grad_for_token(input_to_bert_attn_node, token_idx=i)
        parent_to_max_grad_val_for_all_words.append(parent_to_max_grad_val)

    max_grad_per_parent_per_word: OrderedDict[TokenIndex, Dict[NodeName, float]] = OrderedDict(
        {i: parent_to_max_grad_val_for_all_words[i] for i in range(len(parent_to_max_grad_val_for_all_words))}
    )

    return count_hidden_unit_max_grad_parents, max_grad_per_parent_per_word


def plot_grad_per_parent_per_word_for_all_layers(
    input_to_bert_attn_nodes, tokens, layer_nums=None, remove_mask=False, cut_off_outliers=True
):
    if layer_nums is None:
        layer_nums = range(len(input_to_bert_attn_nodes))
    rows, cols = math.ceil(len(input_to_bert_attn_nodes) / 2), 2
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 16))
    fig.suptitle("Amount of max grad passing through each path in BertAttention Layers")
    for layer_num in layer_nums:
        ax = axes[layer_num // cols, layer_num % cols]
        grad_per_parent_per_word_df = preprocessgrad_per_parent_per_word_data_for_layer(
            input_to_bert_attn_nodes, layer_num=layer_num, remove_mask=remove_mask
        )
        plot_grad_per_parent_per_word_for_layer(
            grad_per_parent_per_word_df,
            tokens,
            layer_num=layer_num,
            ax=ax,
            use_legend=(layer_num == layer_nums[-1]),
            cut_off_outliers=cut_off_outliers,
        )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_grad_per_parent_per_word_for_layer(
    grad_per_parent_per_word_df, tokens, layer_num: int, ax=None, use_legend=False, cut_off_outliers=True
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    # plt.figure(figsize=(10,6))
    sns.lineplot(
        data=grad_per_parent_per_word_df,
        x="token_index",
        y="path grad",
        hue="path_name",
        hue_order=sorted(grad_per_parent_per_word_df["path_name"].unique()),
        legend=use_legend,
        ax=ax,
    )
    # plt.xticks(rotation=45)
    if cut_off_outliers:
        ylim_top = grad_per_parent_per_word_df["path grad"].mean() + 2 * grad_per_parent_per_word_df["path grad"].std()
        ax.set_ylim(bottom=0, top=ylim_top)
    # if grad_per_parent_per_word_df["path grad"]
    ax.set_xticks(ticks=sorted(grad_per_parent_per_word_df["token_index"].unique()), labels=tokens)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(f"Layer {layer_num}", wrap=True)
    # plt.show()
    # return fig


def preprocessgrad_per_parent_per_word_data_for_layer(
    input_to_bert_attn_nodes, tokens, layer_num: int, agg_method: str, remove_mask=False
):
    _, grad_per_parent_per_word = summarize_max_grad_kvq(input_to_bert_attn_nodes[layer_num], agg_method=agg_method)
    grad_per_parent_per_word_df = pd.DataFrame(grad_per_parent_per_word)
    grad_per_parent_per_word_df = grad_per_parent_per_word_df.reset_index().rename(columns={"index": "path_name"})
    grad_per_parent_per_word_df = grad_per_parent_per_word_df.melt(
        id_vars=["path_name"], var_name="token_index", value_name="path grad"
    )  # ["variable"].value_counts()
    grad_per_parent_per_word_df["path grad"] = grad_per_parent_per_word_df["path grad"].astype(float)
    grad_per_parent_per_word_df["path_name"] = grad_per_parent_per_word_df["path_name"].apply(
        lambda x: "skip layer" if "combine" in x else x
    )
    grad_per_parent_per_word_df["layer_num"] = layer_num
    grad_per_parent_per_word_df["token"] = grad_per_parent_per_word_df["token_index"].apply(lambda x: tokens[x])
    if remove_mask:
        grad_per_parent_per_word_df = grad_per_parent_per_word_df[
            grad_per_parent_per_word_df["token_index"] != "[MASK]"
        ]

    return grad_per_parent_per_word_df


def preprocessgrad_per_parent_per_word_data_per_layer(input_to_bert_attn_nodes, tokens, agg_method, remove_mask=False):
    return pd.concat(
        [
            preprocessgrad_per_parent_per_word_data_for_layer(
                input_to_bert_attn_nodes, tokens, layer_num, agg_method=agg_method, remove_mask=remove_mask
            )
            for layer_num in range(len(input_to_bert_attn_nodes))
        ],
        axis=0,
    )


def plot_max_grad_against_layer_per_word(grad_per_parent_per_word_per_layer_df, tokens: List[str], save_path: str):
    # Plot the grad on y axis and layer on x axis for each word in the sentence
    # tokens = grad_per_parent_per_word_per_layer_df["token_index"].unique()
    rows = math.ceil(len(tokens) / 3)
    cols = 3
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(len(tokens), len(tokens)))
    fig.suptitle(
        f"Amount of max grad passing through each path in each BertAttention Layer, for each word\n'{' '.join(tokens)}'\n",
        wrap=True,
    )

    for token_num, token in enumerate(tokens):
        ax = axes[token_num // cols, token_num % cols]
        data = grad_per_parent_per_word_per_layer_df[grad_per_parent_per_word_per_layer_df["token_index"] == token_num]
        use_legend = token_num == len(tokens) - 1
        sns.lineplot(
            data=data,
            x="layer_num",
            y="path grad",
            hue="path_name",
            hue_order=sorted(grad_per_parent_per_word_per_layer_df["path_name"].unique()),
            legend=use_legend,
            ax=ax,
        )
        ax.set_title(token)
        # ax.set_ylim(0, 40)
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    ax.get_legend().remove()
    plt.savefig(save_path)
