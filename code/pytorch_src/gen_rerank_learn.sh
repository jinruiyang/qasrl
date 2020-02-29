#!/bin/bash

source ~/.bashrc
source activate minicondatest
cd /nas/home/starrant/Plan-and-write/pytorch_src/

data_dir="../data/rocstory/splits/ROCStories_titlesepkeysepstory/two_sent_context/"
out_dir="../generation_results/"
model_dir="../models/"
model="ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_511_lr10"
beam=10
scorers="${model_dir}scorer_weights_learn.tsv"



python -m generate.py --learn --lr 1.0 --split_on \| --apply_disc --max_lines 100 --scorers ${scorers} --data ${data_dir}test.txt.delimited --lm ${model_dir}${model}.pt --vocab ${model_dir}${model}.pkl --out ${out_dir}gen_learn.txt --beam_size 5 --print --cuda #> ../logs/${model}_gen_rerank.log

