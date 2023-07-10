# Script to compute MDL of models for synthetic datasets.

import json
import itertools
import random
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from requests.exceptions import HTTPError
import numpy as np
import pandas as pd
import plac
import pytorch_lightning as pl
import sklearn.metrics as metrics
import sys
import torch
import torch.nn as nn
import torchmetrics as tmetrics

import tqdm

from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.callback import Callback
from transformers import BertConfig, BertForSequenceClassification, AdamW
import wandb
from preprocessing.datasets import (
    Contains1,
    FirstTokenRepeatedImmediately,
    FirstTokenRepeatedLast,
    AdjacentDuplicate,
    BinCountOnes,
    ContainsTokenSet,
    ContainsTokenSetOOD,
)
from trainer import BertClassifier
from entropy_utils import validation_with_entropy


@plac.pos("DATASET_NAME", "Name of the dataset class", type=str)
@plac.opt("SEED", "random seed", type=int, abbrev="S")
@plac.opt("SEQLEN", "sequence length", type=int, abbrev="L")
@plac.opt("VOCABSIZE", "vocab size", type=int, abbrev="V")
@plac.opt(
    "NUMPOINTS", "number of total points in dataset (alternative to passing explicit train/val/test sizes)", type=int
)
@plac.opt("TRAINSIZE", "number of train points in dataset", type=int, abbrev="TS")
@plac.opt("VALSIZE", "number of val points in dataset", type=int, abbrev="VS")
@plac.opt("TESTSIZE", "number of test points in dataset", type=int, abbrev="ES")
@plac.opt("NUMCLASSES", "number of bins to divide the dataset", type=int, abbrev="C")
@plac.opt("NUMEPOCHS", "number of epochs to train during computing MDL", type=int, abbrev="NE")
@plac.opt(
    "TOKENSETMAX",
    "When the dataset is ContainsTokenSet, this determines how big the token set is",
    type=int,
    abbrev="TM",
)
@plac.flg("COMPUTEENTROPY", "whether to compute entropy", abbrev="E")
# @plac.flg("OVERWRITE", "whether to overwrite existing saved models and recompute MDL", abbrev="O")
def main(
    DATASET_NAME,
    SEED=1,
    SEQLEN=36,
    VOCABSIZE=60,
    NUMPOINTS=None,
    TRAINSIZE=3600,
    VALSIZE=360,
    TESTSIZE=360,
    NUMCLASSES=2,
    NUMEPOCHS=10,
    COMPUTEENTROPY=False,
    TOKENSETMAX=None,  # inclusive
    # OVERWRITE=False,
):
    """Trains and evaluates model.

    NOTE:
    * If `task` = finetune, then `probe` is ignored.
    * If `task` = probe, then `rate` is ignored.

    NOTE: Use the `properties.py` file to generate your data.
    """
    OVERWRITE = False

    # Set seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Set model and data params
    # DATASET_NAME, DATASET_KWARGS_IDENTIFIABLE = "Contains1", {"num_points": 20000}
    # DATASET_NAME, DATASET_KWARGS_IDENTIFIABLE = "FirstTokenRepeatedImmediately", {
    #     "num_points": 20000
    # }
    # DATASET_NAME, DATASET_KWARGS_IDENTIFIABLE = "FirstTokenRepeatedLast", {
    #     "num_points": 20000
    # }
    # DATASET_NAME, DATASET_KWARGS_IDENTIFIABLE = "AdjacentDuplicate", {"num_points": 20000}

    # TODO: be wary if num classes is not 2 and it's not the bincount problem!! this will default to 2 and you might see silent failures
    DATASET_KWARGS_IDENTIFIABLE = {
        "num_points": NUMPOINTS,
        "vocab_size": VOCABSIZE,
        "train_size": TRAINSIZE,
        "val_size": VALSIZE,
        "test_size": TESTSIZE,
        "seq_len": SEQLEN,
    }

    # Only BinCountOnes allows for multiple classes
    if DATASET_NAME == "BinCountOnes":
        DATASET_KWARGS_IDENTIFIABLE = {**DATASET_KWARGS_IDENTIFIABLE, **{"num_classes": NUMCLASSES}}

    if DATASET_NAME == "ContainsTokenSet" or DATASET_NAME == "ContainsTokenSetOOD":
        if TOKENSETMAX is None or TOKENSETMAX == -1:
            raise ValueError(
                "Dataset ContainsTokenSet requires a valid TOKENSETMAX value (between 1 and seq_len) passed in (received None)."
            )
        DATASET_KWARGS_IDENTIFIABLE = {**DATASET_KWARGS_IDENTIFIABLE, **{"token_set": list(range(1, TOKENSETMAX + 1))}}

    SAMPLING_FRACTION = 1.0
    BATCH_SIZE = 32
    BF_BATCH_SIZE = 4
    HIDDEN_SIZE = 64
    NUM_EPOCHS = NUMEPOCHS
    NUM_ATTENTION_HEADS = 4
    NUM_HIDDEN_LAYERS = 2
    ATTENTION_PROBS_DROPOUT_PROB = 0
    MODEL_CONFIG_PATH = "../brunoflow/models/bert/config-toy.json"
    LEARNING_RATE = 0.001
    L1_WEIGHT = 0.0
    L2_WEIGHT = 0.0
    DROPOUT_PROB = 0

    params_to_log = {k: v for k, v in locals().items() if k.isupper()}

    dataset = getattr(sys.modules[__name__], DATASET_NAME)(**{**DATASET_KWARGS_IDENTIFIABLE})
    data_id = f"{dataset.get_name()}"
    data_dir_no_seed = os.path.join("data", DATASET_NAME, data_id)
    data_dir = os.path.join(data_dir_no_seed, f"{SAMPLING_FRACTION}-{SEED}")

    # Construct model id
    model_id = (
        f"Bert-hs{HIDDEN_SIZE}-numheads{NUM_ATTENTION_HEADS}-bs{BATCH_SIZE}-lr{LEARNING_RATE}-n{NUM_EPOCHS}-nc{SEQLEN}"
    )
    model_id += f"-l1_weight{L1_WEIGHT}" if L1_WEIGHT != 0 else ""
    model_id += f"-l2_weight{L2_WEIGHT}" if L2_WEIGHT != 0 else ""
    model_id += f"-dropoutprob{DROPOUT_PROB}" if DROPOUT_PROB != 0 else ""

    model_dir = os.path.join(data_dir, "models", model_id)
    os.makedirs(model_dir, exist_ok=True)

    model_config_kwargs = dict(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB,
        vocab_size=VOCABSIZE,
    )
    # set the num_labels to the max possible for bincountones (even when num_classes is lower) so that the models have same number of connections regardless of the number of classes.
    model_config_kwargs["num_labels"] = SEQLEN  # if DATASET_NAME == "BinCountOnes" else NUMCLASSES
    accumulate_grad_batches = 1
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using: {accelerator}.")

    # Load data
    train_data, val_data, test_data = load_data(DATASET_NAME, DATASET_KWARGS_IDENTIFIABLE)

    # Load model
    num_steps = (len(train_data) // BATCH_SIZE) * NUM_EPOCHS
    # model = load_model(MODEL_CONFIG_PATH, model_config_kwargs, num_steps)

    params_to_log["num_steps"] = num_steps

    # Init separately so that the nested metrics get logged correctly
    wandb.init(
        project="bauer-bert-synthetic",
        config=params_to_log,
        tags=[
            "compute_mdl",
            "official_result1",
            "mdl_fixed",
            "36classes",
            "log_fine_metrics",
            "ood",
            "log_val_data_and_ood",
        ],
    )

    wandb_logger = WandbLogger()

    save_model_path = os.path.join(model_dir, "model_mdl.pt")

    try:
        artifact_dir = wandb_logger.download_artifact(
            artifact=f"{data_id}-all_models:latest",
            save_dir=data_dir_no_seed,
            use_artifact=True,
        )
        print(f"Downloaded artifact to {artifact_dir}.")
    except (wandb.CommError, HTTPError) as e:
        # HTTPError occurs when the artifact has been deleted, this appears to be a bug with wandb. Catching this exception for now.
        print(type(e), e)

    # Compute MDL and save the model
    if OVERWRITE or not os.path.exists(save_model_path):
        # Model is not cached, recompute MDL and save model.
        additional_results, block_logs, trainer, classifier = compute_mdl(
            train_data,
            MODEL_CONFIG_PATH,
            model_config_kwargs,
            BATCH_SIZE,
            NUM_EPOCHS,
            accumulate_grad_batches,
            num_classes=model_config_kwargs["num_labels"],
            lr=LEARNING_RATE,
            save_model_path=save_model_path,
            logger=wandb_logger,
        )
        print(additional_results)
        wandb.log(additional_results)
    else:
        print("Loading trained model from cache!")
        # Model is cached, skip MDL computation and load model from cache.
        classifier: BertClassifier = load_model(
            model_config_path=MODEL_CONFIG_PATH,
            model_config_kwargs=model_config_kwargs,
            num_steps=num_steps,
            num_classes=model_config_kwargs["num_labels"],
            lr=LEARNING_RATE,
        )
        classifier.encoder.load_state_dict(torch.load(save_model_path))
        trainer = Trainer(
            accelerator=accelerator,
            devices=1,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            min_epochs=NUM_EPOCHS,
            max_epochs=NUM_EPOCHS,
            accumulate_grad_batches=accumulate_grad_batches,
            logger=wandb_logger,
        )

    print(f"Logging model to w&b run {wandb.run}.")
    artifact = wandb.Artifact(name=f"{data_id}-all_models", type="model")
    artifact.add_dir(local_path=data_dir_no_seed)
    wandb.run.log_artifact(artifact)

    # Evaluate the model for accuracy and loss (fast) on the val_dataloader
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    final_model_val_metrics = trainer.test(model=classifier, dataloaders=val_dataloader)
    print(final_model_val_metrics)
    wandb.log(final_model_val_metrics[0])

    # Compute the entropy and things of the model with Brunoflow on the val_dataloader
    val_dataloader = DataLoader(val_data, batch_size=BF_BATCH_SIZE, shuffle=False)
    val_X = torch.stack(list(zip(*val_data))[0], axis=0).numpy().tolist()
    val_labels = torch.stack(list(zip(*val_data))[1], axis=0).numpy().tolist()
    val_X_is_ood = [dataset.is_ood(x) for x in val_X]
    wandb.log(
        {"val_data": json.dumps(val_X), "val_labels": json.dumps(val_labels), "val_X_is_ood": json.dumps(val_X_is_ood)}
    )
    if COMPUTEENTROPY:
        val_metrics = validation_with_entropy(
            classifier.encoder, val_loader=val_dataloader, epoch=NUM_EPOCHS - 1, batch=0, compute_entropy=True
        )
        print(val_metrics)

    wandb.finish()


def load_data(dataset_name, dataset_params):
    """Load data from the IMDB dataset, splitting off a held-out set."""
    dataset = getattr(sys.modules[__name__], dataset_name)(**{**dataset_params})

    train_data = list(dataset.get_train_data())
    # pd.DataFrame([len(set(x.numpy()).intersection(range(1, TOKENSETMAX + 1))) for x in list(zip(*list(dataset.get_train_data())))[0]]).value_counts() # Code to check for the number of examples in each intersect size
    val_data = list(dataset.get_val_data())
    test_data = list(dataset.get_test_data())

    print("train", len(train_data))
    print("val", len(val_data))
    print("test", len(test_data))
    return train_data, val_data, test_data


def load_model(model_config_path: str, model_config_kwargs: dict, num_steps: int, num_classes: int, lr: float):
    """Loads appropriate model & optimizer (& optionally lr scheduler.)

    Parameters
    ----------
    model : ``str``
        model string. in most cases, a hugging face model code.
    num_steps : ``int``
        number of update steps. optionally used for lr schedules.
    """
    return BertClassifier(
        model=None,
        model_config_path=model_config_path,
        model_config_kwargs=model_config_kwargs,
        num_steps=num_steps,
        num_classes=num_classes,
        lr=lr,
    )


def random_split_partition(zipped_list, sizes, num_classes):
    # NOTE: I'm getting some strange issues where the 0.1% doesn't have
    # two labels, thus it gets some bad errors. 0.1% = 0.001, for 2000 * 0.001 = 2,
    # so fair enough.
    # SOLUTION: The training data is shuffled and contains equal counts (or close enough)
    # of labels.
    # TODO: how to compute MDL for large # of classes?
    random.shuffle(zipped_list)
    if num_classes != 2:
        max_label_value = max(
            list(zip(*zipped_list))[1]
        )  # assume that there are labels from 0...max_label_value inclusive
        interleaved_list = list(
            itertools.chain(*zip(*[[z for z in zipped_list if z[1].item() == i] for i in range(max_label_value + 1)]))
        )  # This will result in equally balanced classes but not exactly len(zipped_list) datapoints because it will drop points from each class to make classes exactly balanced (bc of zip)
    else:
        pos = [z for z in zipped_list if z[1].item() in {1, "1", "yes"}]
        neg = [z for z in zipped_list if z[1].item() not in {1, "1", "yes"}]
        interleaved_list = list(itertools.chain(*zip(pos, neg)))
    interleaved_list += list(
        set(zipped_list).difference(set(interleaved_list))
    )  # Add the remaining imbalanced examples to the end to make the number whole
    assert len(interleaved_list) == len(zipped_list)
    return [interleaved_list[end - length : end] for end, length in zip(itertools.accumulate(sizes), sizes)]


def compute_mdl(
    train_data,
    model,
    model_config_kwargs,
    batch_size,
    num_epochs,
    accumulate_grad_batches,
    num_classes,
    lr,
    save_model_path=None,
    logger=None,
):
    """Computes the Minimum Description Length (MDL) over the training data given the model.

    We use *prequential* MDL.

    Voita, Elena, and Ivan Titov. "Information-Theoretic Probing with Minimum Description Length."
    arXiv preprint arXiv:2003.12298 (2020). `https://arxiv.org/pdf/2003.12298`

    Parameters
    ----------
    ``train_data``: list of tuples of examples and labels.
    ``model``: A model string.
    """
    # NOTE: These aren't the split sizes, exactly; the first training size will be the first split size,
    # the second will be the concatenation of the first two, and so on. This is to take advantage
    # of the random_split function.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # split_proportions = np.array([0.1, 0.1, 0.2, 0.4, 0.8, 1.6, 3.05, 6.25, 12.5, 25, 50])
    # split_proportions = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0, 12.0, 14.0, 50.0])
    split_proportions = np.array([1 / 18, 1 / 18, 1 / 9, 1 / 9, 2 / 9, 4 / 9, 1, 2, 2, 4, 8, 16, 16, 50])
    # split_proportions = np.array([100.])
    split_sizes = np.ceil(0.01 * len(train_data) * split_proportions)

    # How much did we overshoot by? We'll just take this from the longest split
    extra = np.sum(split_sizes) - len(train_data)
    split_sizes[len(split_proportions) - 1] -= extra

    splits = random_split_partition(train_data, split_sizes.astype(int).tolist(), num_classes=num_classes)
    mdls = []
    block_logs = []

    # Cost to transmit the first via a uniform code
    mdls.append(
        split_sizes[0] * np.log(num_classes)
    )  # Assume uniform distribution for model, so the entropy is log(num_classes). Then, multiply by the number of examples split_sizes[0] because ??

    for i in tqdm.trange(len(splits), desc="mdl"):
        # If training on the last block, we test on all the data.
        # Otherwise, we train on the next split.
        last_block = i == (len(splits) - 1)

        # setup the train and test split.
        train_split = list(itertools.chain.from_iterable(splits[0 : i + 1]))
        test_split = train_split if last_block else splits[i + 1]

        # re-fresh model.
        datamodule = DataModule(batch_size, train_split, test_split[:batch_size], test_split)
        num_steps = (len(train_split) // batch_size) * num_epochs
        classifier = load_model(model, model_config_kwargs, num_steps, num_classes, lr)
        classifier.extra_name = f"tss-{len(train_split)}_"
        trainer = Trainer(
            accelerator=accelerator,
            devices=1,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            min_epochs=num_epochs,
            max_epochs=num_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            logger=logger,
        )
        trainer.fit(classifier, datamodule=datamodule)

        # Test
        test_result = trainer.test(datamodule=datamodule)
        test_loss = test_result[0][f"{classifier.extra_name}test_loss"]
        block_logs.append(
            {
                "length": len(test_split),
                "loss": test_loss,
            }
        )

        if not last_block:
            mdls.append(test_loss)

    # Save state_dict of model trained on all the data
    torch.save(classifier.encoder.state_dict(), save_model_path)

    total_mdl = np.sum(np.asarray(mdls))
    # the last test_loss is of the model trained and evaluated on the whole training data,
    # which is interpreted as the data_cost
    data_cost = test_loss
    model_cost = total_mdl - data_cost
    return (
        {
            "total_mdl": total_mdl,
            "data_cost": data_cost,
            "model_cost": model_cost,
        },
        block_logs,
        trainer,
        classifier,
    )


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data, val_data, test_data):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


class LossAuc(Callback):
    def __init__(self, monitor="val_loss"):
        super().__init__()
        self.losses = []
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, _):
        if trainer.sanity_checking:
            return
        logs = trainer.callback_metrics
        if self.monitor in logs:
            self.losses.append(logs[self.monitor])

    def get(self):
        if len(self.losses) == 0:
            return 0
        # We assume that the list contains pytorch tensor floats.
        return sum(self.losses).item()


if __name__ == "__main__":
    plac.call(main)
    # main(DATASET_NAME="Contains1", seed=0)
