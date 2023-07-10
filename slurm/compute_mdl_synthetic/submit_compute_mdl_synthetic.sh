#!/bin/bash
cd ../..; # the python scripts are all in the root directory.
python compute_mdl.py $1 --SEED $2 --SEQLEN $3 -V $4 -TS $5 -VS $6 -ES $7 -C $8 -TM $9 -NE ${10} ${11}; # last is whether to compute entropy
