#!/bin/bash
cd ../..; # the python scripts are all in the root directory.
python analyze_kvq.py -i $1 $2 -d "data/lgd_sva/lgd_max_grad_eval_dataset.tsv" -m "google/bert_uncased_L-6_H-512_A-8";
