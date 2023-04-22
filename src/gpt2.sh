#!/bin/bash
#SBATCH --job-name=GPT2-Finetuning
#SBATCH --output=vaspgpt2.out
#SBATCH --error=vaspgpt2.err
#SBATCH --account=eel6825
#SBATCH --qos=eel6825
#SBATCH --mail-type=ALL
#SBATCH --mail-user=v.pathak@ufl.edu
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=2          
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=15gb
#SBATCH --time=24:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

module purge
ml conda

mamba activate llm

T1=$(date +%s)
export MODEL_NAME=gpt2
declare -a task_names=(boolq rte cb wic wsc copa)
for TASK_NAME in "${task_names[@]}"
do   
  python train.py \
  --model_name_or_path gpt2-medium \
  --task_name $TASK_NAME \
  --max_length 1024 \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --seed 42 \
  --weight_decay 0.01 \
  --num_warmup_steps 100 \
  --classifier_dropout 0.1 \
  --clip 1.0 \
  --checkpointing_steps epoch \
  --output_dir /blue/eel6825/v.pathak/experiments/results/$MODEL_NAME/$TASK_NAME/
done
# export TASK_NAME=record
# export TASK_NAME=multirc
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
