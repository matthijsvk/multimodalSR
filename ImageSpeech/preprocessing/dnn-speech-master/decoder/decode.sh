#!/bin/bash

# eval.scp should contains the list of wave files wants to recognize
./recognize_lna_dir.sh lna_eval_estimated eval.scp eval_estimated.trn

./recognize_lna_dir.sh lna_eval_actual eval.scp eval_actual.trn

/work/courses/T/S/89/5150/general/dnn/sclite/sclite -i wsj -f 1 -h eval_estimated.trn -r eval_actual.trn
