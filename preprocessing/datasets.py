import ast
import functools
import os
from tkinter import W
import numpy as np
import pandas as pd
from typing import List, Set
import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from transformers import BertTokenizerFast


class Dataset:
    def __init__(
        self,
        seed,
        num_points=None,
        train_fraction=0.7,
        val_fraction=0.2,
        test_fraction=0.1,
        train_size=None,
        val_size=None,
        test_size=None,
    ) -> None:
        if num_points is not None and train_size is not None:
            raise ValueError("Train/val/test sizes specified by both exact size and fraction! Please just pick one.")
        if num_points is None and train_size is None:
            raise ValueError("Need to provide EITHER exact train size or num_points plus fractions!")
        if train_fraction + val_fraction + test_fraction != 1.0:
            raise ValueError(
                f"Train fraction of {train_fraction}, val_fraction of {val_fraction}, and test_fraction of {test_fraction} does not sum to 1.0."
            )

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.name = None

        self.seed = seed
        self.R = np.random.RandomState(self.seed)
        self.R_torch = torch.Generator().manual_seed(self.seed)

        self.num_points = num_points if num_points is not None else (train_size + val_size + test_size)
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        self.train_size = (
            train_size if train_size is not None else int(self.train_fraction * self.num_points)
        )  # Prefer exact number over fraction if train_size is provided
        self.val_size = val_size if train_size is not None else int(self.val_fraction * self.num_points)
        self.test_size = test_size if train_size is not None else int(self.test_fraction * self.num_points)

    def get_name(self):
        return self.name

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

    def train_val_test_split(self):
        return torch.utils.data.random_split(
            self.full_dataset, [self.train_size, self.val_size, self.test_size], generator=self.R_torch
        )

    def is_ood(self, example):
        # Override this method for OOD datasets.
        return 0


# class OODDataset:
#     def _generate_ood_data(self, num_points):
#         """
#         Method for generating data that includes train data and OOD data.
#         """
#         raise NotImplementedError("Dataset subclassing OODDataset must implement self._generate_ood_data.")


#     def _generate_train_data(self, num_points):
#         """
#         Method for generating train data which excludes some set of OOD inputs.
#         """
#         raise NotImplementedError("Dataset subclassing OODDataset must implement self._generate_train_data.")

#     def _generate_data(self):
#         self.train_data = self._generate_train_data(self.train_size)
#         self.val_data = self._generate_ood_data(self.val_size)
#         self.test_data = self._generate_ood_data(self.test_size)

#     def train_val_test_split(self):
#         # This just makes train, val, and test all follow the same format as in non-OOD datasets (len(self.train_data) == self.train_size)
#         train_data = torch.utils.data.random_split(
#             self.train_data, [self.train_size], generator=self.R_torch
#         )
#         val_data = torch.utils.data.random_split(
#             self.val_data, [self.val_size], generator=self.R_torch
#         )
#         test_data = torch.utils.data.random_split(
#             self.test_data, [self.test_size], generator=self.R_torch
#         )

#         return train_data, val_data, test_data


class BankAbstract(Dataset):
    def __init__(
        self, df_path: str, train_fraction=0.7, val_fraction=0.1, test_fraction=0.2, dtype=torch.float32
    ) -> None:
        if train_fraction + val_fraction + test_fraction != 1.0:
            raise ValueError(
                f"Train fraction of {train_fraction}, val_fraction of {val_fraction}, and test_fraction of {test_fraction} does not sum to 1.0."
            )
        super().__init__()
        self.df_path = df_path
        self.name = self.get_name()
        self.dtype = dtype
        self._generate_data()
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def get_name(self):
        raise NotImplementedError("Override this in concrete child classes of the BankAbstract Dataset.")

    def _generate_data(self):
        # Convert pandas to nparrays
        df = pd.read_csv(self.df_path)
        X = df.drop(["y_yes"], axis=1)
        y = df["y_yes"]
        X = X.to_numpy()  # shape: ((45211, 7)
        y = y.to_numpy()  # shape: (45211,))
        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))

    def train_val_test_split(self):
        train_size = int(self.train_fraction * len(self.full_dataset))
        val_size = int(self.val_fraction * len(self.full_dataset))
        test_size = len(self.full_dataset) - (train_size + val_size)
        return torch.utils.data.random_split(self.full_dataset, [train_size, val_size, test_size])


class Bank7(BankAbstract):
    def __init__(
        self,
        df_path: str = "data/bank/bank-7-features.csv",
        train_fraction=0.7,
        val_fraction=0.01,
        test_fraction=0.29,
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            df_path=df_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            dtype=dtype,
        )

    def get_name(self):
        return "Bank7"


class BankFull(BankAbstract):
    def __init__(
        self,
        df_path: str = "data/bank/bank-full-one-hot.csv",
        train_fraction=0.7,
        val_fraction=0.01,
        test_fraction=0.29,
        dtype=torch.float32,
    ) -> None:
        super().__init__(
            df_path=df_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            dtype=dtype,
        )

    def get_name(self):
        return "BankFull"


class BreastCancer(Dataset):
    def __init__(self, train_fraction=0.7, val_fraction=0.1, test_fraction=0.2, dtype=torch.float32) -> None:
        if train_fraction + val_fraction + test_fraction != 1.0:
            raise ValueError(
                f"Train fraction of {train_fraction}, val_fraction of {val_fraction}, and test_fraction of {test_fraction} does not sum to 1.0."
            )
        super().__init__()
        self.name = "BreastCancer"
        self.dtype = dtype
        self._generate_data()
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        X, y = torch.tensor(data["data"], dtype=self.dtype), torch.tensor(data["target"], dtype=torch.long)
        means = X.mean(dim=1, keepdim=True)
        stds = X.std(dim=1, keepdim=True)
        normalized_X = (X - means) / stds

        self.full_dataset = data_utils.TensorDataset(normalized_X, y)

    def train_val_test_split(self):
        train_size = int(self.train_fraction * len(self.full_dataset))
        val_size = int(self.val_fraction * len(self.full_dataset))
        test_size = len(self.full_dataset) - (train_size + val_size)
        return torch.utils.data.random_split(self.full_dataset, [train_size, val_size, test_size])


class NoisyLinear(Dataset):
    def __init__(
        self,
        num_points=1000,
        noisiness=0.2,
        num_features=4,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.num_features = num_features
        self.noisiness = noisiness
        self.true_weights = np.concatenate(
            [self.R.uniform(low=-1, high=1, size=(self.num_features - 1,)), [1]]
        )  # bias term of 1
        self.name = f"NoisyLinear-num_points{self.num_points}-noisiness{self.noisiness}-numfeats{self.num_features}"
        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        X = self.R.rand(self.num_points, self.num_features)
        noise = self.R.uniform(low=-self.noisiness / 2, high=self.noisiness / 2, size=(self.num_points,))
        y = X @ self.true_weights + noise
        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class LogisticRegressionDataset(Dataset):
    def __init__(
        self,
        num_points=1000,
        noisiness=0.0,
        num_features=4,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.num_features = num_features
        self.noisiness = noisiness
        self.true_weights = np.concatenate([self.R.uniform(low=-1, high=1, size=(self.num_features))])  # bias term of 1
        self.name = f"LogisticRegressionDataset-num_points{self.num_points}-noisiness{self.noisiness}-numfeats{self.num_features}"
        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        X = self.R.rand(self.num_points, self.num_features) * 10 - 5
        # noise = self.R.uniform(low=-self.noisiness / 2, high=self.noisiness / 2, size=(self.num_points,))
        weighted_sum = X @ self.true_weights
        y = ((1 / (1 + np.exp(-weighted_sum))) > 0.5).astype(int)  # Label as 1 if logit > 0.5
        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class LogisticRegressionSameWeightsDataset(Dataset):
    def __init__(
        self,
        num_points=1000,
        noisiness=0.0,
        num_features=4,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.num_features = num_features
        self.noisiness = noisiness
        self.true_weights = np.ones(shape=(self.num_features,))
        self.name = f"LogisticRegressionSameWeightsDataset-num_points{self.num_points}-noisiness{self.noisiness}-numfeats{self.num_features}"
        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        X = self.R.normal(size=(self.num_points, self.num_features))
        # noise = self.R.uniform(low=-self.noisiness / 2, high=self.noisiness / 2, size=(self.num_points,))
        weighted_sum = X @ self.true_weights
        y = ((1 / (1 + np.exp(-weighted_sum))) > 0.5).astype(int)  # Label as 1 if logit > 0.5
        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class LogisticRegressionSameWeightsSampleLabelsDataset(Dataset):
    def __init__(
        self,
        num_points=1000,
        noisiness=0.0,
        num_features=4,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.num_features = num_features
        self.noisiness = noisiness
        self.true_weights = np.ones(shape=(self.num_features,))
        self.name = f"LogisticRegressionSameWeightsDataset-num_points{self.num_points}-noisiness{self.noisiness}-numfeats{self.num_features}"
        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        X = self.R.normal(size=(self.num_points, self.num_features))
        # noise = self.R.uniform(low=-self.noisiness / 2, high=self.noisiness / 2, size=(self.num_points,))
        weighted_sum = X @ self.true_weights
        y = self.R.binomial(n=1, p=(1 / (1 + np.exp(-weighted_sum))))
        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class FirstTokenRepeatedOnce(Dataset):
    """Label is True if first token is repeated at any point, otherwise False"""

    def __init__(
        self,
        vocab_size: int = 20,
        seq_len: int = 10,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.num_features = 4
        self.name = f"FirstTokenRepeatedOnce-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i, first_token in enumerate(X_first_tokens):
            vocab_minus_first_token: list = list(self.eligible_vocab.difference({first_token}))
            X_arr[i, 0] = first_token
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Set random values for all remaining tokens in the sequence that are not the first token
                X_arr[i, 1:] = self.R.choice(vocab_minus_first_token, size=self.seq_len - 1, replace=True)
            else:
                # Label should be True, so repeat the first token at one location in the sequence and
                # assign random (non-first-token) values for the other locations in the sequence.
                # What's the index where the first token will be repeated? (for the examples where there is a repeat)
                repeat_location = self.R.choice(a=np.arange(1, self.seq_len))
                X_arr[i, repeat_location] = first_token
                for j in range(len(X_arr[i])):
                    if j not in {0, repeat_location}:
                        X_arr[i, j] = self.R.choice(vocab_minus_first_token)

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class FirstTokenRepeatedOnceImmediately(Dataset):
    """Label is True if first token is repeated immediately at 2nd position, otherwise False. Also there'll be no other repeats"""

    def __init__(
        self,
        vocab_size: int = 20,
        seq_len: int = 10,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.num_features = 4
        self.name = (
            f"FirstTokenRepeatedOnceImmediately-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"
        )

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i, first_token in enumerate(X_first_tokens):
            vocab_minus_first_token: list = list(self.eligible_vocab.difference({first_token}))
            X_arr[i, 0] = first_token
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Set random values for all remaining tokens in the sequence that are not the first token
                X_arr[i, 1:] = self.R.choice(vocab_minus_first_token, size=self.seq_len - 1, replace=False)
            else:
                # Label should be True, so repeat the first token at one location in the sequence and
                # assign random (non-first-token) values for the other locations in the sequence.
                # What's the index where the first token will be repeated? (for the examples where there is a repeat)
                X_arr[i, 1] = first_token
                X_arr[i, 2:] = self.R.choice(vocab_minus_first_token, size=self.seq_len - 2, replace=False)

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class FirstTokenRepeatedImmediately(Dataset):
    """Label is True if first token is repeated immediately at 2nd position, otherwise False.
    There CAN be repeats here."""

    def __init__(
        self,
        vocab_size: int = 20,
        seq_len: int = 10,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.num_features = 4
        self.name = (
            f"FirstTokenRepeatedImmediately-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"
        )

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i, first_token in enumerate(X_first_tokens):
            vocab_minus_first_token: list = list(self.eligible_vocab.difference({first_token}))
            X_arr[i, 0] = first_token
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Set random values for all remaining tokens in the sequence that are not the first token
                X_arr[i, 1] = self.R.choice(vocab_minus_first_token, replace=False)
                X_arr[i, 2:] = self.R.choice(list(self.eligible_vocab), size=self.seq_len - 2, replace=True)
            else:
                # Label should be True, so repeat the first token at one location in the sequence and
                # assign random (non-first-token) values for the other locations in the sequence.
                # What's the index where the first token will be repeated? (for the examples where there is a repeat)
                X_arr[i, 1] = first_token
                X_arr[i, 2:] = self.R.choice(vocab_minus_first_token, size=self.seq_len - 2, replace=True)

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class FirstTokenRepeatedLast(Dataset):
    """Label is True if first token is repeated immediately at 2nd position, otherwise False. Also there'll be no other repeats"""

    def __init__(
        self,
        vocab_size: int = 20,
        seq_len: int = 10,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.num_features = 4
        self.name = f"FirstTokenRepeatedLast-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i, first_token in enumerate(X_first_tokens):
            vocab_minus_first_token: list = list(self.eligible_vocab.difference({first_token}))
            X_arr[i, 0] = first_token
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Set last token to be not the 1st token
                X_arr[i, -1] = self.R.choice(vocab_minus_first_token, replace=False)
                # Set random values for all remaining tokens in the sequence (repeats of 1st token are ok)
                X_arr[i, 1:-1] = self.R.choice(list(self.eligible_vocab), size=self.seq_len - 2, replace=True)
            else:
                # Label should be True, so repeat the first token at one location in the sequence and
                # assign random values for the other locations in the sequence.
                # What's the index where the first token will be repeated? (for the examples where there is a repeat)
                X_arr[i, -1] = first_token
                X_arr[i, 1:-1] = self.R.choice(list(self.eligible_vocab), size=self.seq_len - 2, replace=True)

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class AdjacentDuplicate(Dataset):
    """Label is True if there are consecutive duplicates anywhere in the string, otherwise False."""

    def __init__(
        self,
        vocab_size: int = 20,
        seq_len: int = 10,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.num_features = 4
        self.name = f"AdjacentDuplicate-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _label(self, x: List[int]) -> bool:
        for i in range(0, len(x) - 1):
            if x[i] == x[i + 1]:
                return True
        return False

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i, first_token in enumerate(X_first_tokens):
            X_arr[i, 0] = first_token
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Handle generation case where the label should be False
                # Randomly generate strings until one doesn't contain all of the tokens in self.token_set
                candidate_false_X = self.R.choice(list(self.eligible_vocab), size=self.seq_len, replace=True)
                while self._label(candidate_false_X):
                    candidate_false_X = self.R.choice(list(self.eligible_vocab), size=self.seq_len, replace=True)
                X_arr[i, :] = candidate_false_X
            else:
                # Randomly choose which index repeat_index will have a repeat at index repeat_index+1 and what the token at those two indices will be
                repeat_index = self.R.choice(range(self.seq_len - 1))
                token_val = self.R.choice(list(self.eligible_vocab))

                # Set the repeat_index and repeat_index+1 to token_val
                X_arr[i, repeat_index] = token_val
                X_arr[i, repeat_index + 1] = token_val

                # Set the other indices to random values (repeats allowed)
                X_arr[i, :repeat_index] = self.R.choice(list(self.eligible_vocab), size=repeat_index, replace=True)
                X_arr[i, repeat_index + 2 :] = self.R.choice(
                    list(self.eligible_vocab), size=self.seq_len - (repeat_index + 2), replace=True
                )

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class CTL(Dataset):
    """Label is True if the sequence contains ALL tokens in the given set"""

    def __init__(
        self,
        ctl_dir,
        seed=0,
    ) -> None:
        super().__init__()

        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.train_path = os.path.join(ctl_dir, "train.csv")
        self.val_path = os.path.join(ctl_dir, "val.csv")
        self.test_path = os.path.join(ctl_dir, "test.csv")

        self.train_data = self._make_tensor_dataset(self.train_path)
        self.val_data = self._make_tensor_dataset(self.val_path)
        self.test_data = self._make_tensor_dataset(self.test_path)

        self.name = "CTL"

    def _make_tensor_dataset(self, df_path: str):
        df = pd.read_csv(df_path)
        tensor_dataset = data_utils.TensorDataset(
            torch.tensor(df["tokenized_input"].apply(ast.literal_eval)), torch.tensor(df["label"])
        )

        return tensor_dataset


class BinCountOnes(Dataset):
    """
    Create num_classes bins by seq_len / num_classes.
    Then, given a sequence, the label is which bin the number of 1s in the sequence falls into.

    Ex:
    Given seq_len = 12, num_classes = 4, then:
        Bin 0 = [1,2,3]
        Bin 1 = [4,5,6]
        Bin 2 = [7,8,9]
        Bin 3 = [10,11,12]

    For a sequence 098765431212, there's 2 1s, so this sequence is labeled 0 (Bin 0).
    Note that all sequences will have at least one 1 so that the classes are counted evenly.
    """

    def __init__(
        self,
        num_classes: int = 2,
        vocab_size: int = 20,
        seq_len: int = 12,
        tokenizer_class=BertTokenizerFast,
        num_points=None,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.num_classes = num_classes
        if seq_len % self.num_classes != 0:
            raise ValueError(
                f"This dataset requires seq_len to be divisible by num_classes. Seq_len of {seq_len} % num_classes of {num_classes} != 0."
            )
        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.name = f"BinCountOnes-num_classes{num_classes}-seqlen{self.seq_len}-num_points{self.num_points}-vs{self.vocab_size}"

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / self.num_classes will be 0, second self.num_points / self.num_classes will be 1, ...
        data_per_class = []
        labels_per_class = []
        if self.num_points % self.num_classes != 0:
            raise ValueError(
                "self.num_classes must evenly divide self.num_points or else some datapoints will be dropped."
            )
        points_per_class = int(self.num_points // self.num_classes)

        vocab_minus_1: list = list(self.eligible_vocab.difference({1}))
        for c in range(self.num_classes):
            labels = np.full(points_per_class, c, dtype=int)
            min_num_ones_in_sequence = c * self.seq_len / self.num_classes + 1
            max_num_ones_in_sequence = (c + 1) * self.seq_len / self.num_classes
            X_arr = np.ones((points_per_class, self.seq_len), dtype=int)
            for p in range(points_per_class):
                num_ones = self.R.randint(min_num_ones_in_sequence, max_num_ones_in_sequence + 1)
                num_not_ones = self.seq_len - num_ones
                X_arr[p, num_ones:] = self.R.choice(vocab_minus_1, size=num_not_ones)
                X_arr[p] = self.R.permutation(X_arr[p])

            data_per_class.append(X_arr)
            labels_per_class.append(labels)

        X = np.concatenate(data_per_class, axis=0)
        y = np.concatenate(labels_per_class, axis=0)

        assert len(X) == len(y)

        # Shuffle the data points so the True and False labels are at random indices.
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class ContainsTokenSet(Dataset):
    """Label is True if the sequence contains ALL tokens in the given set"""

    def __init__(
        self,
        token_set: Set[int],
        vocab_size: int = 20,
        seq_len: int = 15,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.token_set = set(token_set)
        if len(self.token_set) != len(token_set):
            raise ValueError(
                f"Expected token_set {token_set} to contain only unique elements, but set(token_set) has different length from token_set. Ensure your input for token_set has only unique elements."
            )

        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.name = f"ContainsTokenSet-{''.join(str(tok) for tok in self.token_set)}-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        # X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i in range(len(X_arr)):
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Randomly generate strings until one doesn't contain all of the tokens in self.token_set
                candidate_false_X = self.R.choice(list(self.eligible_vocab), size=self.seq_len, replace=True)
                while self.token_set.issubset(set(candidate_false_X)):
                    candidate_false_X = self.R.choice(list(self.eligible_vocab), size=self.seq_len, replace=True)
                X_arr[i, :] = candidate_false_X

            else:
                # Label should be True, so make sure subset is in the string
                X_arr[i, : len(self.token_set)] = list(self.token_set)
                X_arr[i, len(self.token_set) :] = self.R.choice(
                    list(self.eligible_vocab), size=self.seq_len - len(self.token_set), replace=True
                )
                X_arr[i] = self.R.permutation(X_arr[i])

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)

        # Shuffle the data points so the True and False labels are at random indices.
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class ContainsTokenSetOOD(ContainsTokenSet):
    """Label is True if the sequence contains ALL tokens in the given set"""

    def __init__(
        self,
        token_set: Set[int],
        vocab_size: int = 20,
        seq_len: int = 15,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            token_set=token_set,
            vocab_size=vocab_size,
            seq_len=seq_len,
            tokenizer_class=tokenizer_class,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )
        self.name = f"ContainsTokenSetOOD-{''.join(str(tok) for tok in self.token_set)}-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"

    def is_ood(self, example: np.ndarray) -> int:
        """
        Return whether example is OOD of the train set or not as a 0 or 1.
        """
        return int(len(set(example).intersection(self.token_set)) == (len(self.token_set) - 1))

    def _construct_examples_with_overlap(self, num_overlap, num_examples):
        examples = np.zeros(shape=(num_examples, self.seq_len))
        for ex in range(num_examples):
            overlap_nums: set = set(self.R.choice(list(self.token_set), size=num_overlap, replace=False))
            ineligible_nums: set = self.token_set.difference(overlap_nums)
            eligible_vocab = self.eligible_vocab.difference(ineligible_nums)
            examples[ex, :num_overlap] = list(overlap_nums)
            examples[ex, num_overlap:] = self.R.choice(
                list(eligible_vocab), size=self.seq_len - num_overlap, replace=True
            )
            examples[ex] = self.R.permutation(examples[ex])

        return examples

    def _generate_data_balanced_per_intersect_sizes(self, num_points, eligible_intersect_sizes_for_false: Set[int]):
        """
        Args:
            num_points - total number of points in dataset
            eligible_intersect_sizes_for_false - the possible sizes of the intersect between sequence and the self.token_set.
            Largest possible is {0, 1, ..., len(self.token_set) - 1} which has len(self.token_set) number of elements
        """
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(num_points, dtype=int)
        index_threshold_to_false = num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(num_points, self.seq_len), dtype=int)

        # Create True examples (50%)
        X_arr[:index_threshold_to_false, :] = self._construct_examples_with_overlap(
            num_overlap=len(self.token_set), num_examples=index_threshold_to_false
        )

        # Create False examples (50%, evenly divided between eligible_intersect_sizes_for_false overlap in tokens)
        curr_ind = index_threshold_to_false
        num_smaller_intersect_sizes = len(
            eligible_intersect_sizes_for_false
        )  # number of possible intersect sizes between sequence and token_set (where label will be false), which is [0, 1, ..., len(token_set) - 1], inclusive, which is len(token_set) number of possible sizes
        num_examples_per_smaller_intersection_sz_balanced = (
            num_points - index_threshold_to_false
        ) // num_smaller_intersect_sizes  # This is the number of examples (if balanced) per each smaller intersection size
        for intersect_size in eligible_intersect_sizes_for_false:
            # i is the number of intersecting tokens between the sequence and the token set
            X_arr[
                curr_ind : curr_ind + num_examples_per_smaller_intersection_sz_balanced, :
            ] = self._construct_examples_with_overlap(
                num_overlap=intersect_size, num_examples=num_examples_per_smaller_intersection_sz_balanced
            )
            curr_ind += num_examples_per_smaller_intersection_sz_balanced

        # handle extra by randomly choosing the intersection to be < len(token_set) for each remaining individual instance (there should be fewer than len(token_set) * 2 of these)
        for i in range(curr_ind, num_points):
            X_arr[i : i + 1, :] = self._construct_examples_with_overlap(
                num_overlap=self.R.choice(list(eligible_intersect_sizes_for_false)), num_examples=1
            )

        X = X_arr
        assert len(X) == len(y)

        # Shuffle the data points so the True and False labels are at random indices.
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        return data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))

    def _generate_ood_data(self, num_points):
        return self._generate_data_balanced_per_intersect_sizes(
            num_points=num_points, eligible_intersect_sizes_for_false=set(range(len(self.token_set)))
        )

    def _generate_train_data(self, num_points):
        eligible_intersect_sizes_for_false = set(range(len(self.token_set))) - {len(self.token_set) - 1}
        return self._generate_data_balanced_per_intersect_sizes(
            num_points=num_points, eligible_intersect_sizes_for_false=eligible_intersect_sizes_for_false
        )

    def _generate_data(self):
        self.train_data = self._generate_train_data(self.train_size)
        self.val_data = self._generate_ood_data(self.val_size)
        self.test_data = self._generate_ood_data(self.test_size)

        if any(self.is_ood(self.train_data.tensors[0][i].numpy()) for i in range(len(self.train_data.tensors[0]))):
            raise ValueError(
                "Train data contains OOD datapoints according to self.is_ood. Reimplement self._generate_train_data or self.is_ood to ensure train data contains no OOD examples."
            )

    def train_val_test_split(self):
        # This just makes train, val, and test all follow the same format as in non-OOD datasets (len(self.train_data) == self.train_size)
        train_data = torch.utils.data.random_split(self.train_data, [self.train_size], generator=self.R_torch)[0]
        val_data = torch.utils.data.random_split(self.val_data, [self.val_size], generator=self.R_torch)[0]
        test_data = torch.utils.data.random_split(self.test_data, [self.test_size], generator=self.R_torch)[0]

        return train_data, val_data, test_data


class Contains1FirstToken(Dataset):
    """Label is True if the sequence contains the token 1 in the first index"""

    def __init__(
        self,
        vocab_size: int = 20,
        seq_len: int = 10,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.num_features = 4
        self.name = f"Contains1FirstToken-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        # X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i in range(len(X_arr)):
            vocab_minus_1: list = list(self.eligible_vocab.difference({1}))
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Set random (non-1) values for all tokens in the sequence
                X_arr[i, :] = self.R.choice(vocab_minus_1, size=self.seq_len, replace=True)
            else:
                # Label should be True, so make sure 1 is in the string
                X_arr[i, 0] = 1
                X_arr[i, 1:] = self.R.choice(list(self.eligible_vocab), size=self.seq_len - 1, replace=True)

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class Contains1(Dataset):
    """Label is True if the sequence contains the token 1"""

    def __init__(
        self,
        vocab_size: int = 20,
        seq_len: int = 10,
        tokenizer_class=BertTokenizerFast,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.vocab_size = vocab_size
        self.eligible_vocab: set = set(range(1, self.vocab_size))  # reserve 0-token for padding
        self.seq_len = seq_len
        self.tokenizer = tokenizer_class.from_pretrained("bert-base-uncased")
        self.num_features = 4
        self.name = f"Contains1-num_points{self.num_points}-vs{self.vocab_size}-seqlen{self.seq_len}"

        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        # Generate labels - first self.num_points / 2 will be true, second self.num_points / 2 examples will be False
        y = np.zeros(self.num_points, dtype=int)
        index_threshold_to_false = self.num_points // 2
        y[:index_threshold_to_false] = 1

        # Initialize X
        X_arr = np.zeros(shape=(self.num_points, self.seq_len), dtype=int)
        X_str = []

        # What's the first token for each example?
        # X_first_tokens = self.R.choice(a=list(self.eligible_vocab), size=self.num_points, replace=True)

        for i in range(len(X_arr)):
            vocab_minus_1: list = list(self.eligible_vocab.difference({1}))
            if i >= index_threshold_to_false:
                # Handle generation case where the label should be False
                # Set random (non-1) values for all tokens in the sequence
                X_arr[i, :] = self.R.choice(vocab_minus_1, size=self.seq_len, replace=True)
            else:
                # Label should be True, so make sure 1 is in the string
                X_arr[i, 0] = 1
                X_arr[i, 1:] = self.R.choice(list(self.eligible_vocab), size=self.seq_len - 1, replace=True)
                X_arr[i] = self.R.permutation(X_arr[i])

            X_str.append(" ".join(X_arr[i].astype(str)))

        X = X_arr
        # X = np.array(self.tokenizer(X_str)["input_ids"])
        assert len(X) == len(y)
        shuffle_order = self.R.permutation(len(X))
        X, y = X[shuffle_order], y[shuffle_order]

        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class FFOOM(Dataset):
    """FourFeatures, OnlyOneMatters"""

    def __init__(
        self,
        num_points=1000,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        train_size=None,
        val_size=None,
        test_size=None,
        seed=0,
    ) -> None:
        super().__init__(
            seed=seed,
            num_points=num_points,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.num_features = 4
        self.name = f"FFOOM-num_points{self.num_points}"
        self._generate_data()
        self.train_data, self.val_data, self.test_data = self.train_val_test_split()

    def _generate_data(self):
        X = self.R.rand(self.num_points, self.num_features)
        y = (X[:, 0] > 0.5).astype(int)
        self.full_dataset = data_utils.TensorDataset(torch.tensor(X), torch.tensor(y))


class MNIST(Dataset):
    def __init__(self, normalize_params, noisiness=0.0, classes="ALL") -> None:
        super().__init__()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalize_params)])
        self.train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST("data", train=False, transform=transform)
        self.noisiness = noisiness
        self.classes = classes
        self.name = f"MNIST-noisy{self.noisiness}-classes{self.classes}"

        self.filter_train_and_test_by_class_label()
        self.make_train_data_noisy()

    def make_train_data_noisy(self):
        total_num_labels = len(self.train_data.targets)
        label_perm_inds = torch.randperm(total_num_labels)
        num_noisy_labels = int(len(self.train_data.targets) * self.noisiness)
        noisy_label_inds, _ = label_perm_inds[:num_noisy_labels], label_perm_inds[num_noisy_labels:]
        print(f"Setting {len(noisy_label_inds)}/{total_num_labels} to random values")
        noisy_labels = torch.tensor(self.R.randint(0, 9, len(noisy_label_inds)))
        self.train_data.targets[noisy_label_inds] = noisy_labels

    def filter_train_and_test_by_class_label(self):
        if self.classes == "ALL":
            return
        self.filter_train_by_class()
        self.filter_test_by_class()

    def get_mask(self, labels):
        return functools.reduce(lambda a, b: labels == a | labels == b, self.classes)

    def filter_train_by_class(self):
        labels_mask = self.get_mask(self.train_data.targets)
        self.train_data.data = self.train_data.data[labels_mask]
        self.train_data.targets = self.train_data.targets[labels_mask]

    def filter_test_by_class(self):
        labels_mask = self.get_mask(self.test_data.targets)
        self.test_data.data = self.test_data.data[labels_mask]
        self.test_data.targets = self.test_data.targets[labels_mask]
