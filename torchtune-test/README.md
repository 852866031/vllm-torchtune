# Llama-3.1-8B LoRA fine-tuning with torchtune 0.6.1

Fine-tune `meta-llama/Meta-Llama-3.1-8B` with a LoRA adapter (r=16, α=32,
dropout=0.05, target modules `q_proj`/`k_proj`/`v_proj`/`o_proj`) on
`alpaca_1000_p95.txt`. The run is capped at **15 s wall-clock**, the first
**5 s** are dropped as warmup, and the post-warmup **tokens/second** is
printed at the end.

The base model is already on disk at
`/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B`
(HF cache layout, snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`).

## Files in this directory

| File | Purpose |
| --- | --- |
| `alpaca_1000_p95.txt` | Source dataset (one Alpaca-style example per line). |
| `prepare_dataset.py` | Converts the `.txt` into `alpaca_1000_p95.json`. |
| `llama3_8B_lora.yaml` | torchtune recipe config (matches the LoRA spec). |
| `run_timed.py` | Driver: wraps the recipe with a 15 s cap and 5 s warmup. |
| `llama3-toy-lora-ft/` | Reference PEFT adapter — same LoRA config as the yaml. |

## Prereqs

- Linux + one NVIDIA GPU (≥ 24 GB), CUDA 12.4 driver.
- HF token with access to `meta-llama/Meta-Llama-3.1-8B`.
- `conda` installed.

## Steps

### 1. Create the env
```bash
conda create -n tt061 python=3.10 -y
conda activate tt061
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchao==0.9.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install torchtune==0.6.1 "huggingface_hub[cli]" sentencepiece tiktoken blobfile
# torchtune 0.6.1 pulls torchdata==0.11.0 as a transitive dep.
# The HF CLI is `hf` in current huggingface_hub (the old `huggingface-cli` is deprecated).
```

### 2. Fetch missing metadata files
The local snapshot has the safetensors shards but is missing
`original/tokenizer.model`, `config.json`, and `generation_config.json`,
all of which torchtune's HF checkpointer + tokenizer load on startup.
Pull them into the same cache:
```bash
hf auth login   # token with Meta-Llama-3.1-8B access
HF_HUB_CACHE=/home/jiaxuan_chen/scratch hf download \
    meta-llama/Meta-Llama-3.1-8B \
    original/tokenizer.model config.json generation_config.json
```
Confirm:
```bash
SNAP=/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
ls "$SNAP/config.json" "$SNAP/generation_config.json" "$SNAP/original/tokenizer.model"
```
If `hf` lands on a newer snapshot hash, update the two paths in
`llama3_8B_lora.yaml` to match.

### 3. Convert the dataset
```bash
python prepare_dataset.py --in alpaca_1000_p95.txt --out alpaca_1000_p95.json
```

### 4. Run the timed fine-tune
```bash
python run_timed.py llama3_8B_lora.yaml
```

Expected tail:
```
=== Throughput (post-warmup) ===
steps kept       : 7
window           : 9.84 s (from 5.12s to 14.96s of 15s cap)
tokens processed : 14336
tokens / second  : 1456.7
```
(Exact numbers depend on GPU, batch size, and `max_seq_len`.)

## Troubleshooting

- **`tokenizer.model` not found** — step 2 didn't land in the expected snapshot dir. Either re-run with the correct `HF_HUB_CACHE`, or edit `tokenizer.path` in the yaml to point at wherever `original/tokenizer.model` actually ended up.
- **`HFValidationError: gated repo`** — accept the Llama 3.1 license on HF and run `hf auth login` again.
- **CUDA OOM** — in `llama3_8B_lora.yaml` drop `batch_size` to 1 or set `tokenizer.max_seq_len: 512`.
- **`not enough post-warmup samples`** — a single step is taking >5 s. Lower `batch_size` / `max_seq_len`, or bump `WINDOW_SEC` in `run_timed.py`.
