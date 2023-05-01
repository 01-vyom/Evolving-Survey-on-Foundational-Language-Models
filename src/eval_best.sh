#!/bin/bash
#SBATCH --job-name=BestModel-Evaluation 
#SBATCH --output=vaspevalbest.out
#SBATCH --error=vaspbevalbest.err
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
#SBATCH --time=5:00:00


echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

module purge
ml conda

mamba activate llm

T1=$(date +%s)
export MODEL_NAME=deberta
declare -a task_names=(boolq wic wsc)
for TASK_NAME in "${task_names[@]}"
do   
  python eval.py \
  --model_name_or_path /blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/ \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_eval_batch_size 16 \
  --seed 42 \
  --output_dir /blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/
done
export MODEL_NAME=t5
declare -a task_names=(rte cb copa)
for TASK_NAME in "${task_names[@]}"
do   
  python eval.py \
  --model_name_or_path /blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/ \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_eval_batch_size 16 \
  --seed 42 \
  --output_dir /blue/eel6825/<username>/output/$MODEL_NAME/$TASK_NAME/
done
# export TASK_NAME=record
# export TASK_NAME=multirc
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
