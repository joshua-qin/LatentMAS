#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-01:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J medqa_meta_r
#SBATCH -o medqa/slurm_meta_rerun_%j.out
#SBATCH -e medqa/slurm_meta_rerun_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshuaqin@college.harvard.edu

module load python
source activate latentmas
cd /n/home03/jqin/LatentMAS
mkdir -p medqa

python run.py \
  --method latent_mas \
  --model_name "Qwen/Qwen3-4B" \
  --task medqa \
  --prompt hierarchical \
  --max_samples 50 \
  --latent_steps 20 \
  --greedy \
  --seed 42 \
  --generate_bs 1 \
  --use_meta_prompt_generator | tee medqa/meta_hier_lat20_greedy_n50_seed42_rerun.log
