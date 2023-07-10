# Script to evaluate the entropy for LGD's subject-verb agreement dataset.
# Adapted from https://github.com/yoavg/bert-syntax.
# coding=utf-8
import argparse
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import brunoflow as bf
from collections import Counter
from transformers import BertForMaskedLM, BertTokenizerFast, BfBertForMaskedLM
import torch
import numpy as np
import sys
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Tuple, List, Union, Optional
import wandb
from datetime import datetime
import pandas as pd


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


class GradientAnalyzer:
    def __init__(self, model: Union[BertForMaskedLM, BfBertForMaskedLM]) -> None:
        self.model = model

    def _gather_grads_of_input_ids(
        self, input_ids: List[int], normalize: bool = True, multiply_by_inputs: bool = True
    ) -> np.ndarray:
        """
        After backprop is called, use this to get the gradients for each input_id in the sequence
        (aggregated by summing gradients across the hidden/embedding size).

        Args:
            input_ids: the input ids for each token in the sequence/tokenized sentence.
            normalize: whether to normalize the grads by the norm of the embedding vector when aggregating grads across the embedding
            multiply_by_inputs: whehther to multiply the grad by the embedding value

        Returns:
            np.ndarray with shape = (len(input_ids),)
        """
        raise NotImplementedError(
            "Implement this in a downstream class to handle either torch.Tensors or jnp DeviceArrays"
        )

    def _exec_backward_pass(self, quantity_of_interest):
        """Call backprop or backward depending on if using BF or Torch."""
        raise NotImplementedError("Implement this in a downstream class to handle either torch or bf")

    def compute_grads_of_input_ids(
        self, quantity_of_interest: Union[torch.Tensor, bf.Node], input_ids: List[int], normalize=True
    ) -> np.ndarray:
        """
        Execute the backward pass on the quantity of interest and then compute the gradient for each token/input_id in the input sequence.

        Args:
            quantity_of_interest - the scalar value on which to backprop. Typically the logit_score(correct_word) - logit_score(incorrect_word)
            input_ids - a tuple contaning the input ids for each token in sequence/tokenized sentence.

        Returns:
            np.ndarray with shape = (len(input_ids),)
        """
        # Call backprop on the quantity of interest
        self.model.train()
        self._exec_backward_pass(quantity_of_interest)
        self.model.eval()
        # quantity_of_interest.visualize(collapse_to_modules=True)

        # Inspect the grads of the embeddings of the words in the sentence
        token_grads = self._gather_grads_of_input_ids(input_ids, normalize=normalize)

        return token_grads

    def compute_entropies_of_input_ids(
        self,
        quantity_of_interest: Union[torch.Tensor, bf.Node],
        input_ids,
    ) -> np.ndarray:
        """
        Execute the backward pass on the quantity of interest and then compute the gradient for each token/input_id in the input sequence.

        Args:
            quantity_of_interest - the scalar value on which to backprop. Typically the logit_score(correct_word) - logit_score(incorrect_word)
            input_ids - a tuple contaning the input ids for each token in sequence/tokenized sentence.

        Returns:
            np.ndarray with shape = (len(input_ids),)
        """
        raise NotImplementedError("Implement in subclass.")

    def plot_tokens_and_grads(
        self,
        tokens: Tuple[str],
        token_grads: Tuple[float],
        sent: str = None,
        save_path: str = None,
        show_fig: bool = False,
    ):
        _, ax = plt.subplots(figsize=(10, 8))
        sns.lineplot(token_grads, ax=ax)
        ax.set_title(f"Gradients for each token in sentence: \n{sent if sent is not None else ''}", wrap=True)
        ax.set_xlabel("Token")
        ax.set_ylabel("Gradient")

        for i in range(len(tokens)):
            ax.text(x=i, y=token_grads[i], s=tokens[i], color="green" if token_grads[i] > 0 else "red")

        if save_path is not None:
            plt.savefig(save_path)

        if show_fig:
            plt.show()

        plt.close("all")


class TorchGradientAnalyzer(GradientAnalyzer):
    def __init__(self, model: BertForMaskedLM) -> None:
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                f"TorchGradientAnalyzer expected a Torch Module to be passed in, instead received {type(model)}."
            )
        super().__init__(model)

    def _gather_grads_of_input_ids(
        self, input_ids: List[int], normalize: bool = True, multiply_by_inputs=True
    ) -> np.ndarray:
        grads_word_ids = self.model.get_input_embeddings().weight.grad[
            input_ids
        ]  # shape=(len(input_ids), hidden_sz=128) - the grad of (logit(correct_word) - logit(incorrect+word)) w.r.t. each word in the sentence. Since hidden_sz is the embedding_sz, you get a grad for each bit of the embedding
        if multiply_by_inputs:
            grads_word_ids *= self.model.get_input_embeddings().weight[input_ids]
        token_grads = (
            grads_word_ids.sum(dim=-1).detach().numpy()
        )  # shape = (len(input_ids),) (we aggregated the gradients across all hidden_states for each input id in the sequence)
        if normalize:
            token_grads = (token_grads - token_grads.mean()) / torch.linalg.norm(grads_word_ids).detach().numpy()

        return token_grads

    def _exec_backward_pass(self, quantity_of_interest: torch.Tensor):
        quantity_of_interest.backward()

    def compute_grads_of_input_ids(
        self, quantity_of_interest: torch.Tensor, input_ids: List[int], **kwargs
    ) -> np.ndarray:
        if not isinstance(quantity_of_interest, torch.Tensor):
            raise ValueError(
                f"TorchGradientAnalyzer.compute_grads_of_input_ids expected a torch Tensor to be passed in, instead received {type(quantity_of_interest)}."
            )
        return super().compute_grads_of_input_ids(quantity_of_interest, input_ids)


def _compute_qoi(
    input_ids: List[int],
    word_ids: Tuple[int],
    target_idx: int,
    bert: Union[bf.net.Network, torch.nn.Module],
    use_bf: bool,
):
    scores = compute_scores_for_word_ids(
        input_ids=input_ids, word_ids=word_ids, target_idx=target_idx, bert=bert, use_bf=use_bf
    )
    return compute_score_diff_between_correct_and_incorrect_words(scores)


class BfGradientAnalyzer(GradientAnalyzer):
    def __init__(self, model: BfBertForMaskedLM) -> None:
        if not isinstance(model, bf.net.Network):
            raise ValueError(
                f"BfGradientAnalyzer expected a BF Network to be passed in, instead received {type(model)}."
            )

        super().__init__(model)

    def _gather_grads_of_input_ids(self, input_ids: List[int], normalize: bool = True, multiply_by_inputs: bool = True):
        grads_word_ids = self.model.get_input_embeddings().weight.grad[
            jnp.array(input_ids)
        ]  # shape=(len(input_ids), hidden_sz=128) - the grad of (logit(correct_word) - logit(incorrect+word)) w.r.t. each word in the sentence. Since hidden_sz is the embedding_sz, you get a grad for each bit of the embedding
        if multiply_by_inputs:
            grads_word_ids *= self.model.get_input_embeddings().weight.val[jnp.array(input_ids)]
        token_grads = grads_word_ids.sum(
            axis=-1
        ).__array__()  # shape = (len(input_ids),) (we aggregated the gradients across all hidden_states for each input id in the sequence)

        if normalize:
            token_grads = (token_grads - token_grads.mean()) / jnp.linalg.norm(grads_word_ids).__array__()

        return token_grads

    def _gather_entropies_of_input_ids(self, input_ids: List[int]):
        unnorm_entropies_word_ids = self.model.get_input_embeddings().weight.entropy_wrt_output[
            jnp.array(input_ids)
        ]  # shape=(len(input_ids), hidden_sz=128)

        abs_val_grads_word_ids = self.model.get_input_embeddings().weight.abs_val_grad[
            jnp.array(input_ids)
        ]  # shape=(len(input_ids), hidden_sz=128)

        # Pretend we aggregate across the hidden dim by summing, so effectively for each token we have a new node with an edge to each of the 128 hidden units
        # The entropy semiring product of (1, -log(1)) * (abs_val_grad, entropy_wrt_out) = (abs_val_grad, entropy_wrt_out)
        # So then we just need to sum across the 128, and normalize with those new values
        unnorm_entropies_word_ids_reduced = unnorm_entropies_word_ids.sum(axis=-1)  # shape = (len(input_ids),)
        abs_val_grads_word_ids_reduced = abs_val_grads_word_ids.sum(axis=-1)  # shape = (len(input_ids),)

        token_entropies = unnorm_entropies_word_ids_reduced / abs_val_grads_word_ids_reduced + jnp.log(
            abs_val_grads_word_ids_reduced
        )

        return token_entropies.__array__()

    def _exec_backward_pass(self, quantity_of_interest: bf.Node, values_to_compute=("max_grad",)):
        quantity_of_interest.backprop(values_to_compute=values_to_compute)

    def compute_grads_of_input_ids(self, quantity_of_interest: bf.Node, input_ids: List[int], **kwargs) -> np.ndarray:
        if not isinstance(quantity_of_interest, bf.Node):
            raise ValueError(
                f"BfGradientAnalyzer.compute_grads_of_input_ids expected a bf.Node to be passed in, instead received {type(quantity_of_interest)}."
            )
        return super().compute_grads_of_input_ids(quantity_of_interest, input_ids)

    def compute_entropies_of_input_ids(
        self,
        quantity_of_interest: Union[torch.Tensor, bf.Node],
        input_ids: List[int],
    ) -> np.ndarray:
        """
        Execute the backward pass on the quantity of interest and then compute the gradient for each token/input_id in the input sequence.

        Args:
            quantity_of_interest - the scalar value on which to backprop. Typically the logit_score(correct_word) - logit_score(incorrect_word)
            input_ids - a tuple contaning the input ids for each token in sequence/tokenized sentence.

        Returns:
            np.ndarray with shape = (len(input_ids),)
        """
        if not isinstance(quantity_of_interest, bf.Node):
            raise ValueError(
                f"BfGradientAnalyzer.compute_entropies_of_input_ids expected a bf.Node to be passed in, instead received {type(quantity_of_interest)}."
            )
        # Call backprop on the quantity of interest
        self.model.train()
        self._exec_backward_pass(quantity_of_interest, values_to_compute=("entropy", "abs_val_grad"))
        self.model.eval()
        # quantity_of_interest.visualize(collapse_to_modules=True)

        # Inspect the grads of the embeddings of the words in the sentence
        token_entropies = self._gather_entropies_of_input_ids(input_ids)

        return token_entropies


def compute_scores_for_word_ids(
    input_ids: List[int],
    word_ids: Tuple[int],
    target_idx: int,
    bert: Union[bf.net.Network, torch.nn.Module],
    use_bf: bool,
) -> List[float]:
    """
    Args:
        Sent - the sentence of interest, e.g.
            "a 12th-century commentary on periegetes by eustathius of thessalonica also ***mask*** the shape of konesar malai to a phallus ."
        w1 - the correct word, e.g. "compares"
        w2 - the incorrect word, e.g. "compare"

    Returns:
    Tuple of:
        word_probs - the logit scores for w1 and w2. List of two floats, e.g. [12.930843353271484, 9.259199142456055]
        tokens_and_grads - a list of 2-element tuples of length=tokenized(sent).
            The first element is the token (str).
            The second element is d(score(w1) - score(w2))/d(token); that is, the grad (float) of that token.
    """
    if (isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2) or (
        isinstance(input_ids, bf.Node) and len(input_ids.shape) == 2
    ):
        tens = input_ids
    else:
        tens = torch.LongTensor(input_ids).unsqueeze(0)
    if use_bf:
        tens = bf.Node(jnp.array(tens.numpy()), name="inputs")
    res = bert(
        tens
    ).logits  # shape=(bs=1, seq_len, vs=30522) (essentially, for each index in the sequence, what's the logit for each possible predicted word, of which there are vocab_size options)

    res_for_target_idx = res[
        :, target_idx
    ]  # shape=(vs=30522,) (for the index we care about, what are the logits for all possible classes=vocab?)
    # res=torch.nn.functional.softmax(res,-1)
    scores = res_for_target_idx[
        :, word_ids
    ]  # shape=(2,) (what's the logits for the grammatically correct and incorrect words?)
    return scores


def compute_score_diff_between_correct_and_incorrect_words(scores):
    # assert len(scores) == 2
    return scores[:, 0] - scores[:, 1]  # shape=(), because it's a single value


def get_probs_for_words(
    sent: str,
    w1: str,
    w2: str,
    bert: Union[bf.net.Network, torch.nn.Module],
    tokenizer: BertTokenizerFast,
    use_bf: bool,
    grad_analyzer: Optional[GradientAnalyzer] = None,
    compute_entropy=False,
) -> Tuple[List[float], List[str], List[float]]:
    """
    Args:
        Sent - the sentence of interest, e.g.
            "a 12th-century commentary on periegetes by eustathius of thessalonica also ***mask*** the shape of konesar malai to a phallus ."
        w1 - the correct word, e.g. "compares"
        w2 - the incorrect word, e.g. "compare"

    Returns:
    Tuple of:
        word_probs - the logit scores for w1 and w2. List of two floats, e.g. [12.930843353271484, 9.259199142456055]
        tokens_and_grads - a list of 2-element tuples of length=tokenized(sent).
            The first element is the token (str).
            The second element is d(score(w1) - score(w2))/d(token); that is, the grad (float) of that token.
    """
    tokens, target_idx = convert_sentence_to_tokens_and_target_idx(sent, tokenizer=tokenizer)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    try:
        word_ids = tokenizer.convert_tokens_to_ids([w1, w2])
    except KeyError:
        print("skipping", w1, w2, "bad wins")
        return None

    scores = compute_scores_for_word_ids(
        input_ids=input_ids, word_ids=word_ids, target_idx=target_idx, bert=bert, use_bf=use_bf
    )
    score_diff = compute_score_diff_between_correct_and_incorrect_words(scores)  # shape=(), because it's a single value

    token_grads = (
        grad_analyzer.compute_grads_of_input_ids(
            quantity_of_interest=score_diff,
            input_ids=input_ids,
            word_ids=word_ids,
            target_idx=target_idx,
            bert=bert,
            use_bf=use_bf,
        )
        if grad_analyzer is not None
        else None
    )

    token_entropies = (
        grad_analyzer.compute_entropies_of_input_ids(
            quantity_of_interest=score_diff,
            input_ids=input_ids,
        )
        if grad_analyzer is not None and compute_entropy
        else None
    )

    word_probs = [float(x) for x in scores[0]] if not use_bf else [float(x) for x in list(scores[0].val.__array__())]

    return word_probs, tokens, token_grads, token_entropies


def eval_lgd(
    bert: Union[bf.net.Network, torch.nn.Module],
    tokenizer: BertTokenizerFast,
    grad_analyzer: GradientAnalyzer,
    use_bf: bool,
    model_dir: str,
    plot_token_grads: bool = False,
    dataset_path="data/lgd_sva/lgd_eval_entropy_dataset.tsv",
    compute_entropy=True,
    index_range=None,
):
    results_dir = os.path.join(model_dir, "results", "entropy")

    os.makedirs(results_dir, exist_ok=True)

    DATASET_SUFFIX = ""
    if "entropy" in dataset_path:
        DATASET_SUFFIX = "_entropy"
    elif "max_grad" in dataset_path:
        DATASET_SUFFIX = "_max_grad"

    filename = (
        f"lgd_results_base{DATASET_SUFFIX}_{index_range[0]}_{index_range[1]}.csv"
        if index_range is not None
        else "lgd_results_base.csv"
    )
    results_path = os.path.join(results_dir, filename)

    for i, line in enumerate(open(dataset_path, encoding="utf8")):
        if index_range is not None:
            if i < index_range[0]:
                continue
            if i >= index_range[1]:
                break

        na, _, masked, good, bad = line.strip().split("\t")
        word_probs, tokens, grads, entropies = get_probs_for_words(
            sent=masked,
            w1=good,
            w2=bad,
            bert=bert,
            tokenizer=tokenizer,
            use_bf=use_bf,
            grad_analyzer=grad_analyzer,
            compute_entropy=compute_entropy,
        )

        if word_probs is None:
            continue

        if plot_token_grads:
            plots_dir = os.path.join(results_dir, "plots", "grad_vs_tokens")
            os.makedirs(plots_dir, exist_ok=True)

            assert grads is not None
            grad_analyzer.plot_tokens_and_grads(
                tokens=tokens,
                token_grads=grads,
                sent=masked,
                save_path=os.path.join(plots_dir, f"{i}.png"),
                show_fig=False,
            )
        gp = word_probs[0]
        bp = word_probs[1]
        results = {
            "i": i,
            "correct": gp > bp,
            "na": na,
            "good": good,
            "gp": gp,
            "bad": bad,
            "bp": bp,
            "token": tokens,
            "entropies": entropies,
            "masked": masked.encode("utf8"),
        }
        wandb.log(results)
        df = pd.DataFrame(results)
        df.to_csv(results_path, mode="a", header=False)

        # with open(results_path, "a") as f:
        #     print(
        #         i,
        #         str(gp > bp),
        #         entropies.tostring(),
        #         na,
        #         good,
        #         gp,
        #         bad,
        #         bp,
        #         masked.encode("utf8"),
        #         list(zip(tokens, grads)),
        #         sep="\t",
        #         file=f,
        #     )
        #     if i % 100 == 0:
        #         print(i, file=sys.stderr)
        #         sys.stdout.flush()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--MODEL_ID", type=str, default="google/bert_uncased_L-6_H-512_A-8")
    parser.add_argument("-c", "--CHECKPOINT_ID", type=str, default=None)
    parser.add_argument("-i", "--EXAMPLE_INDICES_RANGE", nargs="+", type=int, default=[0, 10])
    parser.add_argument("-d", "--DATASET_PATH", type=str, default="data/lgd_sva/lgd_dataset.tsv")
    parser.add_argument("--USE_BF", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--COMPUTE_ENTROPY", default=True, action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    args = get_args()

    PROJECT_NAME = "eval_bert_entropy"

    # PARAMETERS
    MODEL_ID = args.MODEL_ID
    CHECKPOINT_ID = args.CHECKPOINT_ID
    EXAMPLE_INDICES_RANGE = args.EXAMPLE_INDICES_RANGE
    DATASET_PATH = args.DATASET_PATH
    DATA_DIR = "data/lgd_sva"
    model_name = MODEL_ID

    params_to_log = vars(args)

    wandb.init(
        project=PROJECT_NAME,
        name=f"eval_bert_entropies_{EXAMPLE_INDICES_RANGE}_{datetime.now().isoformat(sep='_', timespec='seconds')}",
        config=params_to_log,
        tags=["analysis"],
    )

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    USE_BF = args.USE_BF
    if USE_BF:
        bert = BfBertForMaskedLM.from_pretrained(model_name)
        model_name = "bf-" + model_name
        grad_analyzer = BfGradientAnalyzer(bert)
    else:
        bert = BertForMaskedLM.from_pretrained(model_name)
        grad_analyzer = TorchGradientAnalyzer(bert)

    MODEL_DIR = os.path.join(DATA_DIR, model_name)
    if CHECKPOINT_ID is not None:
        CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints", CHECKPOINT_ID)
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_ID}.ckpt")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))["state_dict"]
        renamed_state_dict = {".".join(k.split(".")[1:]): v.cpu() for k, v in state_dict.items()}
        bert.load_state_dict(renamed_state_dict)
        MODEL_DIR = CHECKPOINT_DIR

    PLOT_TOKEN_GRADS = "ptg" in sys.argv

    print(f"using model: {model_name} with model type: {bert._get_name()}", model_name, file=sys.stderr)
    bert.eval()

    eval_lgd(
        bert=bert,
        tokenizer=tokenizer,
        grad_analyzer=grad_analyzer,
        use_bf=USE_BF,
        model_dir=MODEL_DIR,
        plot_token_grads=PLOT_TOKEN_GRADS,
        dataset_path=DATASET_PATH,
        compute_entropy=args.COMPUTE_ENTROPY,
        index_range=EXAMPLE_INDICES_RANGE,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
