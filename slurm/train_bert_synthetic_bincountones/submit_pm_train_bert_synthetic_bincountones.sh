#!/bin/bash
cd ../..; # the python scripts are all in the root directory.
python pm_train_mlp.py -p train_bert_synthetic -d BinCountOnes -s $1 -hs $2 -dp $3 -k '{"seq_len": '$4', "num_points": '$5', "num_classes": '$6'}' -lr 0.001 -n 30 -t bincountones entropy_mdl slurm -o;
