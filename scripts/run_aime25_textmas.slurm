#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-04:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J aime25_txt
#SBATCH -o aime2025/slurm_textmas_%j.out
#SBATCH -e aime2025/slurm_textmas_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshuaqin@college.harvard.edu

module load python
source activate latentmas
cd /n/home03/jqin/LatentMAS
mkdir -p aime2025

python run.py \
  --method text_mas \
  --model_name "Qwen/Qwen3-4B" \
  --task aime2025 \
  --prompt hierarchical \
  --max_samples 30 \
  --max_new_tokens 20000 \
  --greedy \
  --seed 42 \
  --generate_bs 1 | tee aime2025/textmas_hier_greedy_n30_seed42.log
