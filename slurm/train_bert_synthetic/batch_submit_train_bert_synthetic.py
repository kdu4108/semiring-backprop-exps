import subprocess

seeds = [0, 1, 2, 3, 4]  # val every 10
# seeds = [5, 6, 7, 8, 9] # val every 5
# seeds = [10,11,12,13,14] # val every 2
dataset_names = ["Contains1", "FirstTokenRepeatedImmediately", "FirstTokenRepeatedLast", "AdjacentDuplicate"]
for seed in seeds:
    for dataset_name in dataset_names:
        subprocess.check_call(["sbatch", "submit_pm_train_bert_synthetic.cluster", f"{dataset_name}", f"{seed}"])
