#!/bin/bash

# run from root dir, in an active environment with requirements.txt installed
exp_num=$RANDOM
# Make sure the below works with your folder structure!
data_dir="../QA-SRL/data/seq2seq/" # this is where your conditional data lives
cond_data="test_QA.txt" # this file is the conditional data to use
model_dir="../QA-SRL/model/seq2seq/" # this is where your models live
#plot_temp=$1 # this is temperature for plot sampling
#num_examples=$1 # this is number of examples to generate. If you want the whole file, delete "max_lines" or make this number arbitrarily large.
base_dir="../QA-SRL/output/"
mkdir ${base_dir}
### FILL OUT THE BELOW

model="${model_dir}seq2seq_glove_0227.pt"
vocab="${model_dir}qasrl.pkl"
out_file="${base_dir}test_out_${exp_num}.txt"


### Generate Sentences ###
echo "Generating Sentence..."

python ../QA-SRL/code/pytorch_src/generate.py --vocab ${vocab} --lm ${model} \
--decode_type sample --data ${data_dir}${cond_data}  \
--out ${out_file} --temp 0.8  \
--sep "</s>"
# --max_lines 100
# --nucleus --top_p 0.9 --top_k 5
# --max_lines 10
# --print_cond_data
