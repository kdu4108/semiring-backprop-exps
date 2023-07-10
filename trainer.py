# Helper classes defining Pytorch lightning classifiers

import itertools
import random
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from transformers import BertConfig, BertForSequenceClassification, BertModel, AdamW
import wandb
from preprocessing.datasets import (
    Contains1,
    FirstTokenRepeatedImmediately,
    FirstTokenRepeatedLast,
    AdjacentDuplicate,
    BinCountOnes,
)


class BaseClassifier(pl.LightningModule):
    """The structure for the various modules is largely the same.
    For some models the step, forward, and configure_optimizers functions
    will have to be over-ridden.
    """

    def __init__(self, num_classes=2, lr=0.001):
        super(BaseClassifier, self).__init__()
        self.lr = lr
        self.val_acc = tmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = tmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = tmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = tmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.extra_name = ""

    def step(self, batch):
        device = self.encoder.device
        tokenized, labels = batch
        tokenized = tokenized.to(device)
        # tokenized = self.tokenizer.batch_encode_plus(
        #     texts, add_special_tokens=True, return_tensors="pt", padding=True
        # )["input_ids"].to(device)
        encoded = self.encoder(
            tokenized,
            labels=labels.to(device),
            return_dict=True,
        )
        return encoded.logits, encoded.loss

    def forward(self, batch):
        logits, _ = self.step(batch)
        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        _, loss = self.step(batch)
        self.log(f"{self.extra_name}train_loss", loss, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        _, labels = batch
        logits, loss = self.step(batch)
        pred = torch.argmax(logits, dim=-1)
        self.val_acc(pred, labels)
        self.log(f"{self.extra_name}val_acc", self.val_acc, on_step=True, on_epoch=False)
        self.val_f1(pred, labels)
        self.log(f"{self.extra_name}val_f1", self.val_f1, on_step=True, on_epoch=False)
        self.log(f"{self.extra_name}val_loss", loss, on_step=True, on_epoch=False)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        _, labels = batch
        logits, _ = self.step(batch)
        pred = torch.argmax(logits, dim=-1)
        loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
        acc = self.test_acc(pred, labels)
        # self.log("test_acc", self.test_acc, on_step=True, on_epoch=False)
        f1 = self.test_f1(pred, labels)
        # self.log("test_f1", self.test_f1, on_step=True, on_epoch=False)
        # self.log("test_loss", loss, on_step=True, on_epoch=False)
        results = {
            f"{self.extra_name}test_loss": loss,
            f"{self.extra_name}test_acc": acc,
            f"{self.extra_name}test_f1": f1,
        }
        self.log_dict(results)
        return results


class BertClassifier(BaseClassifier):
    def __init__(
        self,
        model: BertModel,
        model_config_path: str,
        model_config_kwargs: dict,
        num_steps: int,
        num_classes=2,
        lr=0.001,
    ):
        if model is not None and model_config_path is not None:
            raise ValueError(
                "Either one of `model` or `model_config_path` must be None. Use `model` to pass in an instantiated BertModel/ForSeqClassification/etc as the encoder, or `model_config_path` (+ `model_config_kwargs`) to specify a pretrained model to load."
            )
        super(BertClassifier, self).__init__(num_classes, lr=lr)
        self.encoder = self.make_model_from_config(
            model_config_path, model_config_kwargs
        )  # BertForSequenceClassification.from_pretrained(model)
        self.num_steps = num_steps

    def make_model_from_config(self, model_config_path: str, model_config_kwargs: dict):
        """Loads appropriate model & optimizer (& optionally lr scheduler.)

        Parameters
        ----------
        model : ``str``
            model string. in most cases, a hugging face model code.
        num_steps : ``int``
            number of update steps. optionally used for lr schedules.
        """
        config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=model_config_path,
            **model_config_kwargs,
        )

        model = BertForSequenceClassification(config=config)
        return model
