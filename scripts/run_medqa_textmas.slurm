#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-03:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J medqa_textmas
#SBATCH -o medqa/slurm_textmas_%j.out
#SBATCH -e medqa/slurm_textmas_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshuaqin@college.harvard.edu

module load python
source activate latentmas
cd /n/home03/jqin/LatentMAS
mkdir -p medqa

python run.py \
  --method text_mas \
  --model_name "Qwen/Qwen3-4B" \
  --task medqa \
  --prompt hierarchical \
  --max_samples 50 \
  --greedy \
  --seed 42 \
  --generate_bs 1 | tee medqa/textmas_hier_greedy_n50_seed42.log
