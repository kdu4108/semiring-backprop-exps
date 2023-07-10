import subprocess

seeds = [0, 1, 2, 3, 4]
num_classes = [2, 3, 4, 6, 9, 12, 18, 36]
num_points = [1200, 3600, 10800]
seq_lens = [36]
h_sizes = [64]
dropouts = [
    0.0
]  # https://api.wandb.ai/links/kdu/cgzwj062 - dropout prob doesn't make a difference for accuracy and entropy for bincountones task
# seeds = [6, 7, 8, 9, 10]
# num_classes = [2, 3, 4, 6, 12]
for seed in seeds:
    for hs in h_sizes:
        for d in dropouts:
            for seq_len in seq_lens:
                for np in num_points:
                    for nc in num_classes:
                        subprocess.check_call(
                            [
                                "sbatch",
                                "submit_pm_train_bert_synthetic_bincountones.cluster",
                                f"{seed}",
                                f"{hs}",
                                f"{d}",
                                f"{seq_len}",
                                f"{np}",
                                f"{nc}",
                            ]
                        )
