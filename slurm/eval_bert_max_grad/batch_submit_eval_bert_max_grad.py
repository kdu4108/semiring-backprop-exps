import subprocess

num_points = 1000
num_parallel = 200
num_points_per_instance = int(num_points // num_parallel)
for i in range(0, num_parallel):
    subprocess.check_call(
        [
            "sbatch",
            "submit_eval_bert_max_grad.cluster",
            f"{i * num_points_per_instance}",
            f"{(i+1) * num_points_per_instance}",
        ]
    )
