#!/bin/bash
cd ../..; # the python scripts are all in the root directory.
python eval_bert.py -i $1 $2 -d "data/lgd_sva/lgd_entropy_eval_dataset.tsv" -m "google/bert_uncased_L-6_H-512_A-8" -c $3;
