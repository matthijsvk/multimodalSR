#!/bin/bash


[ "$#" -eq 3 ] || exit "3 arguments required: input_dir scp_list out_file"

input_dir=$1
scp_list=$2
out_file=$3

export MODEL_DIR=/work/courses/T/S/89/5150/general/dnn/models
export AM="${MODEL_DIR}/base_monophone"
export LM="${MODEL_DIR}/morph40000_6gram_gt_1e-8"
export LOOKAHEAD_LM="${MODEL_DIR}/morph40000_2gram_gt"
export CLASSES=
export FSA=0
export SPLIT_MULTIWORDS=
export BEAM=180
export LM_SCALE=30
export TOKEN_LIMIT=10000

export PYTHONPATH="/work/courses/T/S/89/5150/general/dnn/decoder-lib"

/work/courses/T/S/89/5150/general/dnn/recognize.py --bin-lm ${LM}.bin --am $AM --dictionary ${MODEL_DIR}/morph40000_mono.lex --beam $BEAM --language-model-scale $LM_SCALE --token-limit 100000 --hypothesis-file $out_file --work-directory $input_dir --rec-directory $input_dir -f $scp_list
