#!/bin/bash

source ~/.bashrc
source activate minicondatest
cd /nas/home/starrant/Plan-and-write/pytorch_src/

data_dir="../data/rocstory/splits/ROCStories_titlesepkeysepstory/two_sent_context/"
out_dir="../generation_results/"
model_dir="../models/"
model="ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_511_lr10"
beam=10
scorers="${model_dir}scorer_weights_low.tsv"



python -m generate.py --both --apply_disc --print_cond_data --scorers ${scorers} --data ${data_dir}test.txt.context --lm ${model_dir}${model}.pt --vocab ${model_dir}${model}.pkl --out ${out_dir}all_scorers_0.1.txt --beam_size 10 --cuda --print > ../logs/all_scorers_0.1.log

