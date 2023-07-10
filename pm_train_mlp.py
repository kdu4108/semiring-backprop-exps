# Papermill runner script for executing experiments written as jupyter notebooks

import argparse
import datetime
import json
import papermill as pm
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--DATASET_NAME", type=str, default="MNIST")
    parser.add_argument("-s", "--SEEDS", nargs="+", type=int, default=[0])
    parser.add_argument("-bs", "--BATCH_SIZE", type=int, default=32)
    parser.add_argument("-is", "--INPUT_SIZE", type=int)
    parser.add_argument("-hs", "--HIDDEN_SIZES", nargs="+", type=int, default=[16])
    parser.add_argument("-n", "--NUM_EPOCHS", type=int, default=1)
    parser.add_argument("-lr", "--LEARNING_RATE", type=float)
    parser.add_argument("-l1", "--L1_WEIGHT", type=float, default=0)
    parser.add_argument("-l2", "--L2_WEIGHT", type=float, default=0)
    parser.add_argument("-dp", "--DROPOUT_PROBS", nargs="+", type=float, default=[0])
    parser.add_argument("-t", "--TAGS", nargs="+", type=str)
    parser.add_argument("-o", "--OVERWRITE_MODEL", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-v", "--VALIDATE_DURING_TRAINING", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-k", "--DATASET_KWARGS_IDENTIFIABLE", type=json.loads)
    parser.add_argument("-p", "--NOTEBOOK_NAME", type=str, default="train_mnist_mlp")

    return parser.parse_args()


def get_pm_params(
    args: dict, excluded_args: set = {"SEEDS", "HIDDEN_SIZES", "DROPOUT_PROBS", "NOTEBOOK_NAME"}, extra_params=dict()
) -> dict:
    """Return the dict consisting of only the parameters we want Papermill to inject into the template notebook."""
    return {**{k: v for k, v in args.items() if k not in excluded_args and v is not None}, **extra_params}


def main():
    """The executor for a fleet of experiments, written in a Jupyter notebook. We use Papermill to execute notebooks and for code provenance."""
    args = get_args()
    template_notebook_path = os.path.join(os.getcwd(), f"{args.NOTEBOOK_NAME}.ipynb")
    pm_out_dir = os.path.join(os.getcwd(), f"pm_{args.NOTEBOOK_NAME}_outputs")
    os.makedirs(pm_out_dir, exist_ok=True)

    for SEED in args.SEEDS:
        for HIDDEN_SIZE in args.HIDDEN_SIZES:
            for DROPOUT_PROB in args.DROPOUT_PROBS:
                PM_RUN_ID = f"{args.DATASET_NAME}_{SEED}_hs{HIDDEN_SIZE}_bs{args.BATCH_SIZE}_lr{args.LEARNING_RATE}_l1weight{args.L1_WEIGHT}_l2weight_{args.L2_WEIGHT}_dropoutprob_{DROPOUT_PROB}_{datetime.datetime.now().isoformat(sep='_', timespec='seconds')}.ipynb"
                output_notebook_path = os.path.join(pm_out_dir, PM_RUN_ID)
                pm_params = get_pm_params(
                    vars(args),
                    extra_params={
                        "SEED": SEED,
                        "HIDDEN_SIZE": HIDDEN_SIZE,
                        "DROPOUT_PROB": DROPOUT_PROB,
                        "PM_RUN_ID": PM_RUN_ID,
                    },
                )

                print(f"Executing notebook {template_notebook_path} with params {pm_params}")
                pm.execute_notebook(template_notebook_path, output_notebook_path, parameters=pm_params)


if __name__ == "__main__":
    main()
