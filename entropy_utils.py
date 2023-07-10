# Helper script containing shared functions to compute entropy of a model.

import json
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax import numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import numpy as np
import torch
from typing import List, Union
import wandb

import brunoflow as bf
from brunoflow.opt import Adam, cross_entropy_loss, regularize
from transformers import BfBertForSequenceClassification


def convert_torch_to_bf_model(model: torch.nn.Module):
    """Load a BF model from a saved model path (either torch or bf)"""
    bf_model = BfBertForSequenceClassification(model.config)
    bf_model.load_state_dict(model.cpu().state_dict())
    return bf_model


def validation_with_entropy(
    model: Union[torch.nn.Module, bf.net.Network],
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    batch: int,
    compute_entropy=False,
    max_val_set_size=None,
):
    model = convert_torch_to_bf_model(model)
    optimizer = Adam(model.parameters(), step_size=0.001)  # LR doesn't matter, we just need to zero grad

    # Initialize accumulators (bc we will need to break validation into many batches for computing entropy, etc)
    total_correct = 0
    total_loss = 0
    total_test_points = 0
    total_entropy = 0
    total_grad = 0
    total_abs_val_grad = 0

    is_correct = []
    # losses = []
    entropies = []
    grads = []
    abs_val_grads = []

    model.eval()

    # Loop through each batch in the val_loader
    for inputs, labels in val_loader:
        inputs = inputs.numpy()
        labels = labels.numpy()

        # Apply model to inputs
        bert_outputs = model(inputs, labels=labels)
        logits = bert_outputs.logits
        is_correct_in_batch = np.argmax(logits.val, axis=1) == labels
        num_correct_in_batch = sum(is_correct_in_batch)
        loss: bf.Node = cross_entropy_loss(logits, labels, reduction="sum")

        if compute_entropy:
            # Compute and accumulate entropy, grad, abs_val_grad for the batch in the validation set
            optimizer.zero_gradients()
            model.train()
            loss.backprop(values_to_compute=("abs_val_grad", "entropy", "grad"))
            model.eval()
            entropy_per_example_per_token: np.ndarray = gather_entropies_of_input_ids(
                model=model, input_ids=inputs
            )  # shape: (bs, seq_len)

            abs_val_grads_per_example_per_token = gather_abs_grad_of_input_ids(
                model, inputs
            )  # shape=(len(input_ids), hidden_sz)
            grads_per_example_per_token = gather_grad_of_input_ids(model, inputs)  # shape=(len(input_ids), hidden_sz)

            entropy = np.sum(entropy_per_example_per_token)  # shape: ()
            abs_val_grad = np.sum(abs_val_grads_per_example_per_token)
            grad = np.sum(grads_per_example_per_token)

            total_entropy += entropy
            total_abs_val_grad += abs_val_grad
            total_grad += grad

            entropies += entropy_per_example_per_token.tolist()
            abs_val_grads += abs_val_grads_per_example_per_token.tolist()
            grads += grads_per_example_per_token.tolist()
            # entropy_per_example = np.mean(entropy_per_example_per_token, axis=1) # we want a (len(val_loader), seq_len) array
            # entropies += entropy
            # abs_val_grads.append(abs_val_grad)
            # grads.append(grad)
        is_correct += is_correct_in_batch.astype(int).tolist()

        total_loss += loss
        total_correct += num_correct_in_batch
        total_test_points += len(labels)
        if max_val_set_size is not None and total_test_points > max_val_set_size:
            break

    # Compute mean statistics across entire validation cohort
    accuracy = total_correct / total_test_points
    mean_entropy = total_entropy / total_test_points if compute_entropy else None
    mean_grad = total_grad / total_test_points if compute_entropy else None
    mean_abs_val_grad = total_abs_val_grad / total_test_points if compute_entropy else None
    mean_loss = total_loss / total_test_points

    val_metrics = {
        "val": {
            "loss": mean_loss.val,
            "grad": mean_grad,
            "abs_grad": mean_abs_val_grad,
            "entropy": mean_entropy,
            "epoch": epoch,
            "batch": batch,
            "accuracy": accuracy,
            "is_correct_per_example": json.dumps(is_correct),
            "grads_per_example_per_token": json.dumps(grads),
            "abs_val_grads_per_example_per_token": json.dumps(abs_val_grads),
            "entropies_per_example_per_token": json.dumps(entropies),
        }
    }

    if wandb.run is not None:
        wandb.log(val_metrics)

    model.train()

    return val_metrics


def gather_entropies_of_input_ids(model, input_ids: List[int]):
    unnorm_entropies_word_ids = model.get_input_embeddings().weight.entropy_wrt_output[
        jnp.array(input_ids)
    ]  # shape=(len(input_ids), hidden_sz)

    abs_val_grads_word_ids = model.get_input_embeddings().weight.abs_val_grad[
        jnp.array(input_ids)
    ]  # shape=(len(input_ids), hidden_sz)

    # Pretend we aggregate across the hidden dim by summing, so effectively for each token we have a new node with an edge to each of the 128 hidden units
    # The entropy semiring product of (1, -log(1)) * (abs_val_grad, entropy_wrt_out) = (abs_val_grad, entropy_wrt_out)
    # So then we just need to sum across the hidden_sz, and normalize with those new values
    unnorm_entropies_word_ids_reduced = unnorm_entropies_word_ids.sum(axis=-1)  # shape = (len(input_ids),)
    abs_val_grads_word_ids_reduced = abs_val_grads_word_ids.sum(axis=-1)  # shape = (len(input_ids),)

    token_entropies = unnorm_entropies_word_ids_reduced / abs_val_grads_word_ids_reduced + jnp.log(
        abs_val_grads_word_ids_reduced
    )

    return token_entropies.__array__()  # shape: (len(input_ids),)


def gather_abs_grad_of_input_ids(model, input_ids: List[int]):
    abs_val_grads_word_ids = model.get_input_embeddings().weight.abs_val_grad[
        jnp.array(input_ids)
    ]  # shape=(len(input_ids), hidden_sz)

    # Pretend we aggregate across the hidden dim by taking the mean.
    return abs_val_grads_word_ids.mean(axis=-1).__array__()  # shape: (len(input_ids),)


def gather_grad_of_input_ids(model, input_ids: List[int]):
    grads_word_ids = model.get_input_embeddings().weight.grad[jnp.array(input_ids)]  # shape=(len(input_ids), hidden_sz)

    # Pretend we aggregate across the hidden dim by taking the mean.
    return grads_word_ids.mean(axis=-1).__array__()  # shape: (len(input_ids),)
