#!/bin/bash
#SBATCH --job-name=ROBERTA-Finetuning
#SBATCH --output=vasproberta.out
#SBATCH --error=vasproberta.err
#SBATCH --account=eel6825
#SBATCH --qos=eel6825
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<username>@ufl.edu
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
export MODEL_NAME=roberta
declare -a task_names=(boolq rte cb wic wsc copa)
for TASK_NAME in "${task_names[@]}"
do   
  python train.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --seed 42 \
  --weight_decay 0.1 \
  --checkpointing_steps epoch \
  --output_dir /blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/
done
# export TASK_NAME=record
# export TASK_NAME=multirc
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
