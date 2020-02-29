#!/bin/sh

# default args 1000 1500 0.2 0.1
#Generate a Random Experiment Number
exp_num=$RANDOM

lr=10
model="seq2seq_glove_${exp_num}"
vocab="qasrl"
filebase="/content/gdrive/My\ Drive/QA-SRL/data/seq2seq"
model_dir="/content/gdrive/My\ Drive/QA-SRL/model/seq2seq"
log_dir="/content/gdrive/My\ Drive/log/seq2seq"
#corpus="qasrl.pkl"
corpus="/content/gdrive/My\ Drive/QA-SRL/model/corpus.qasrl.pkl.data"

#if use the refactor version then have to use --not-tied

python /content/QA-SRL/code/seq2seq/seq2seq/main.py --seq2seq --attn general --batch-size 32 \
--emsize $1 --nhid $2 --pretrained-name glove \
--train-data ${filebase}/train1000.txt \
--valid-data ${filebase}/dev1000.txt \
--test-data ${filebase}/test.txt \
--dropouti $3 --dropouth $4 --seed 141 \
--epoch 20 --lr ${lr} \
--vocab-file ${model_dir}/${vocab}.pkl --save ${model_dir}/${model}.pt \
--delimiter '<EQA>' --keep-delimiter \
--when 4 8 --cuda

#--cached-data ${model_dir}/${corpus}

#--resume ${model_dir}${model}.pt

#> ${log_dir}${model}.log
#--cached-data ${corpus}



