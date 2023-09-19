#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N impli           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -t 1-18
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l gpu_memory=48G

module load miniconda
conda activate peft
mkdir -p logs
DATAPATH="/projectnb/llamagrp/feyzanb/llama"



cnt=0
lr=0.0001
for dataset_name in "creak_n10" "creak_n100" "creak_n1000"; do
for type in "original" "contradiction_implication_filtered" "implication_filtered" "contradiction" "contradiction_implication_unfiltered" "implication_unfiltered"; do
    (( cnt++ ))
    creak_dev="$DATAPATH/creak_dev.csv"
    creak_test="$DATAPATH/$dataset_name/rel_dev.csv"
    creak_train="$DATAPATH/$dataset_name/creak_train_${type}.csv"
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        python finetune.py \
        --creak_train $creak_train \
        --creak_dev $creak_dev \
        --creak_test $creak_test \
        --lr $lr \
        --epoch $epoch \
        --dataset_name $dataset_name_$type > logs/log_${dataset_name}_type_${type}_sampler.txt 2>&1
    fi
done
done



# for debugging

# dataset_name="creak_n50"
# type=contradiction
# creak_dev="$DATAPATH/creak_dev.csv"
# creak_train="$DATAPATH/$dataset_name/creak_train_${type}.csv"
# python clm.py \
#         --creak_train $creak_train \
#         --creak_dev $creak_dev \
#         --lr $lr \
#         --dataset_name $dataset_name_$type
