import subprocess

num_points = 1000
num_parallel = 500
num_points_per_instance = int(num_points // num_parallel)
checkpoint_ids = [
    "epoch=0-step=10000",
    "epoch=1-step=20000",
    "epoch=2-step=30000",
    "epoch=3-step=40000",
    "last",
]
for c in checkpoint_ids:
    for i in range(0, num_parallel):
        subprocess.check_call(
            [
                "sbatch",
                "submit_eval_bert_entropy.cluster",
                f"{i * num_points_per_instance}",
                f"{(i+1) * num_points_per_instance}",
                c,
            ]
        )
