#!/bin/bash

source ~/.bashrc
source activate minicondatest
cd /nas/home/starrant/Plan-and-write/pytorch_src/

data_dir="../data/rocstory/splits/ROCStories_titlesepkeysepstory/"
out_dir=${data_dir} #"../data/rocstory/splits/ROCStories_titlesepkeysepstory/"
model_dir="../models/"
model="ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_511_lr10"
beam=10

for input in disc_train.txt valid.txt test.txt
do
    python generate.py --data ${data_dir}${input}.context --lm ${model_dir}${model}.pt --vocab ${model_dir}${model}.pkl --out ${out_dir}${input}.generated_continuation --beam_size 10 --cuda > ../logs/${model}_${input}.log
done
