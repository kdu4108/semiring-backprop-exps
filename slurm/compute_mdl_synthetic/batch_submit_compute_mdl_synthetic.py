import subprocess

RUN_LOCALLY = True

# # BinCountOnes
# seeds = [0, 1, 2]  # val every 10
# dataset_names = ["BinCountOnes"]
# seq_lens = [36]
# vocab_sizes = [60]
# train_val_test_sizes = [(3600, 360, 360)]
# tokensetmaxes = [-1]
# num_classes = [2, 3, 4, 6, 9, 12, 18, 36]
# num_epochs = 50

# # Non BinCountOnes
# seeds = [0, 1, 2]  # val every 10
# # dataset_names = ["AdjacentDuplicate"]
# dataset_names = ["Contains1", "FirstTokenRepeatedImmediately", "FirstTokenRepeatedLast", "AdjacentDuplicate"]
# seq_lens = [36]
# vocab_sizes = [60]
# train_val_test_sizes = [(3600, 360, 360)]
# tokensetmaxes = [-1]
# num_classes = [2]
# num_epochs = 50

# # ContainsTokenSet
# seeds = [0, 1, 2]  # val every 10
# dataset_names = ["ContainsTokenSet"]
# seq_lens = [36]
# vocab_sizes = [60]
# train_val_test_sizes = [(3600, 360, 360)]
# tokensetmaxes = [1, 5, 10, 15, 20, 25, 30, 35]
# num_classes = [2]
# num_epochs = 50

# ContainsTokenSet
seeds = [0, 1, 2]  # val every 10
dataset_names = ["ContainsTokenSetOOD"]
seq_lens = [36]
vocab_sizes = [60]
train_val_test_sizes = [(3600, 360, 360)]
tokensetmaxes = [1, 5, 10, 15, 20, 25, 30, 35]
num_classes = [2]
num_epochs = 50

compute_entropy = False
for seed in seeds:
    for dataset_name in dataset_names:
        for seq_len in seq_lens:
            for vocab_sz in vocab_sizes:
                for ts, vs, es in train_val_test_sizes:
                    for nc in num_classes:
                        for tsm in tokensetmaxes:
                            if RUN_LOCALLY:
                                subprocess.run(
                                    [
                                        "python",
                                        "compute_mdl.py",
                                        f"{dataset_name}",
                                        "-S",
                                        f"{seed}",
                                        "-L",
                                        f"{seq_len}",
                                        "-V",
                                        f"{vocab_sz}",
                                        "-TS",
                                        f"{ts}",
                                        "-VS",
                                        f"{vs}",
                                        "-ES",
                                        f"{es}",
                                        "-C",
                                        f"{nc}",
                                        "-TM",
                                        f"{tsm}",
                                        "-NE",
                                        f"{num_epochs}",
                                    ]
                                    + (["-E"] if compute_entropy else [])
                                )
                            else:
                                subprocess.check_call(
                                    [
                                        "sbatch",
                                        "submit_compute_mdl_synthetic.cluster",
                                        f"{dataset_name}",
                                        f"{seed}",
                                        f"{seq_len}",
                                        f"{vocab_sz}",
                                        f"{ts}",
                                        f"{vs}",
                                        f"{es}",
                                        f"{nc}",
                                        f"{tsm}",
                                        f"{num_epochs}",
                                    ]
                                    + (["-E"] if compute_entropy else [])
                                )
