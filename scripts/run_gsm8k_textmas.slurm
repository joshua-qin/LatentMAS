#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-03:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J gsm8k_txt
#SBATCH -o gsm8k/slurm_textmas_%j.out
#SBATCH -e gsm8k/slurm_textmas_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshuaqin@college.harvard.edu

module load python
source activate latentmas
cd /n/home03/jqin/LatentMAS
mkdir -p gsm8k

python run.py \
  --method text_mas \
  --model_name "Qwen/Qwen3-4B" \
  --task gsm8k \
  --prompt hierarchical \
  --max_samples 50 \
  --max_new_tokens 2048 \
  --greedy \
  --seed 42 \
  --generate_bs 1 | tee gsm8k/textmas_hier_greedy_n50_seed42.log
