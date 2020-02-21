#!/bin/sh

# default args 1000 1500 0.2 0.1
#Generate a Random Experiment Number
exp_num=$RANDOM

lr=10
model="seq2seq_glove_${exp_num}"
vocab="qasrl"
filebase="../QA-SRL/data/seq2seq"
model_dir="../QA-SRL/model/seq2seq"
log_dir="../QA-SRL/log/seq2seq"
corpus="corpus.qasrl.pkl.data"

#if use the refactor version then have to use --not-tied

python code/seq2seq/seq2seq/main.py --seq2seq --attn general --batch-size 64 \
--emsize $1 --nhid $2 --pretrained-name glove \
--train-data ${filebase}/train.txt \
--valid-data ${filebase}/dev.txt \
--test-data ${filebase}/test.txt \
--dropouti $3 --dropouth $4 --seed 141 \
--epoch 20 --lr ${lr} \
--vocab-file ${model_dir}/${vocab}.pkl --save ${model_dir}/${model}.pt \
--delimiter '<EQA>' --keep-delimiter \
--when 4 8 --cached-data ${model_dir}/${corpus}

#--resume ${model_dir}${model}.pt

#> ${log_dir}${model}.log
#--cached-data ${corpus}



