#!/bin/bash
cd ../..; # the python scripts are all in the root directory.
python pm_train_mlp.py -p train_bert_synthetic -hs 32 -d $1 -s $2 -k '{"num_points": 40000}' -n 101 -lr 0.001 -t entropy_mdl slurm ev10 -ev 10;
