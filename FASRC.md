# Running LatentMAS on FASRC (first time)

Short guide to run the MedQA control and treatment jobs on Harvard FASRC.

## 1. One-time setup

### 1.1 Request an interactive session (to build the env)

Do **not** run heavy jobs on login nodes. Get a short interactive session (e.g. on `test` partition) to create the environment:

```bash
salloc -p test -t 0-02:00 --mem=8G -c 2
```

Once you get a node, continue below. Use `exit` when done.

### 1.2 Load Python and create the environment

```bash
module load python
# Create env with Python 3.10 (as in README)
mamba create -n latentmas python=3.10 -y
source activate latentmas
```

### 1.3 Install dependencies

From the repo root (`LatentMAS/`):

```bash
cd /n/home03/jqin/LatentMAS
pip install -r requirements.txt
```

(Using `pip` **inside** the mamba env is fine per FASRC docs.)

### 1.4 (Optional) Hugging Face cache

To avoid repeated downloads and to use fast storage:

```bash
# Use scratch for large cache (adjust lab name if needed)
export HF_HOME=/n/netscratch/$(groups | awk '{print $1}')/jqin/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME
```

Add these `export` and `mkdir` lines to your `~/.bashrc` if you want them in every session.

### 1.5 Create log directory

```bash
mkdir -p /n/home03/jqin/LatentMAS/medqa
```

---

## 2. Running the two jobs

You need **GPU** nodes (run.py uses `cuda` by default). Use either **interactive** or **batch**. For **Option B (batch)**, the one-time setup in Section 1 must be done first so that the `latentmas` environment and its packages exist.

### Option A: Interactive (good for first run / debugging)

Request a GPU node and run the commands by hand:

```bash
# Request 1 GPU, 4 hours, 32GB RAM (adjust time/mem as needed)
salloc -p gpu -t 0-04:00 --mem=32G -c 8 --gres=gpu:1
```

When you get a node:

```bash
module load python
source activate latentmas
cd /n/home03/jqin/LatentMAS

# Optional: set HF cache if you did in 1.4
# export HF_HOME=... etc.

# Control (no meta prompt generator)
python run.py \
  --method latent_mas \
  --model_name "Qwen/Qwen3-4B" \
  --task medqa \
  --prompt hierarchical \
  --max_samples 50 \
  --latent_steps 20 \
  --greedy \
  --seed 42 \
  --generate_bs 1 | tee medqa/control_hier_lat20_greedy_n50_seed42.log

# Treatment (meta prompt generator on)
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
  --use_meta_prompt_generator | tee medqa/meta_hier_lat20_greedy_n50_seed42.log
```

Then `exit` to leave the allocation.

### Option B: Batch (submit and forget)

**Requirement:** You must have done the **one-time setup (Section 1)** first. Batch jobs do *not* install packages—they only run `module load python` and `source activate latentmas`, so the `latentmas` env (with all dependencies already installed) must exist on the cluster.

From a **login node** (no need to `salloc`):

```bash
cd /n/home03/jqin/LatentMAS
sbatch scripts/run_medqa_control.slurm
sbatch scripts/run_medqa_meta.slurm
```

Check queue: `squeue -u $USER`  
Inspect logs: `medqa/control_hier_lat20_greedy_n50_seed42.log` and `medqa/meta_hier_lat20_greedy_n50_seed42.log`, plus Slurm output/error files (see script headers).

---

## 3. Partitions (reminder)

- **`gpu`**: A100 GPUs (e.g. `--gres=gpu:1`), 3-day max.
- **`gpu_test`**: Short GPU test jobs, 12-hour max, limits apply.
- **`test`**: CPU only, for building env / light work; no GPU.

Use `spart` to see your allowed partitions.

---

## 4. Troubleshooting

- **“No module named …”**  
  Make sure you’re in the same env in the job as when you installed: `module load python` and `source activate latentmas` in the SLURM script or interactive session.

- **Batch job fails but interactive works**  
  Don’t submit from inside a conda/mamba env; run `source deactivate` (or open a new terminal) and then `sbatch ...`. Also avoid `conda initialize` or `source activate` in `~/.bashrc` for batch; activate only inside the job script.

- **Out of memory**  
  Increase `#SBATCH --mem` in the script or request more GPUs if the model doesn’t fit.

- **Hugging Face download**  
  First run may download the model; ensure `HF_HOME` (or default cache) is writable and has space. Using `/n/netscratch/...` is recommended for large caches.
