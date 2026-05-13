# vllm-test — timeline benchmark on vLLM 0.18.0 + LoRA

A minimal reimplementation of `examples/auto_benchmark.py` against a plain
vLLM 0.18.0 OpenAI-compatible server with a single LoRA adapter loaded.

The orchestrator (`run_benchmark.py`) launches the server (`launch_vllm.py`)
in a subprocess, waits for `/health`, runs a warmup phase that replays the
first few timeline rows on schedule, then fires the full timeline. Per-request
metrics are written to a CSV with the same columns as
`examples/example_output.csv`:

```
idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s
```

## 1. Environment setup

The steps below start from a clean conda install and pin vLLM 0.18.0. vLLM
0.18.0 requires Python 3.10–3.12 and CUDA 12.x; pick a Python version that
matches your toolchain (3.11 is a safe default).

```bash
# Create the env
conda create -n vllm-test python=3.11 -y
conda activate vllm-test

# Build tools + base scientific deps
pip install --upgrade pip setuptools wheel

# vLLM 0.18.0 — the published wheel bundles a matching torch + CUDA runtime.
# If you have a custom CUDA install and want vLLM to use your system torch,
# install torch first and add --no-deps to the vllm install.
pip install vllm==0.18.0

# Client-side deps used by run_benchmark.py
pip install aiohttp tqdm
```

Verify the install:

```bash
python -c "import vllm; print(vllm.__version__)"   # -> 0.18.0
vllm serve --help | head -n 3
```

If `vllm serve` cannot find your GPU, double-check `nvidia-smi` and that
`CUDA_VISIBLE_DEVICES` is set correctly.

## 2. What's in this directory

```
vllm-test/
├── README.md             — this file
├── launch_vllm.py        — launches `vllm serve` with the LoRA loaded
├── run_benchmark.py      — orchestrator: spawn server, warmup, timeline, CSV
├── llama3-toy-lora-ft/   — LoRA adapter (target modules q/k/v/o_proj, r=16)
└── examples/
    ├── example_timeline.csv — sample input timeline (timestamps + prompts)
    ├── example_output.csv   — sample output CSV (target format)
    ├── auto_benchmark.py    — original benchmark this mirrors
    └── launch_llama3.py     — original launcher this mirrors
```

The LoRA's `adapter_config.json` lists `meta-llama/Meta-Llama-3-8B` as its
base. We serve it on Meta-Llama-3.1-8B — same architecture, same target
modules (`q,k,v,o_proj`), same rank — so the adapter loads cleanly. The
generations will not match what the adapter was trained for, but the
benchmark only measures latency/TTFT/TBT, not output quality.

## 3. Run the benchmark

Defaults are wired up for this machine:

- base model: `/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- LoRA dir:   `./llama3-toy-lora-ft`
- timeline:   `./examples/example_timeline.csv`
- output:     `./output/timeline_results.csv`

```bash
cd /home/jiaxuan_chen/vllm+torchtune/vllm-test
python run_benchmark.py
```

You'll see two log streams interleaved:

- `[server] ...`     — stdout/stderr of `vllm serve`
- `[orchestrator]`   — warmup/timeline phase markers
- `[warmup]`         — every 10th warmup request
- `Timeline: ...`    — tqdm bar tracking wall-clock vs. timeline duration

When it's done, results land at `./output/timeline_results.csv`.

### Useful flags

```bash
# point at a different timeline / output
python run_benchmark.py \
    --timeline_csv path/to/timeline.csv \
    --out_csv path/to/results.csv

# different port (server side and client side)
python run_benchmark.py --port 9100

# tune the warmup window
python run_benchmark.py --warmup_count 50 --warmup_duration_s 20 --warmup_rest_s 5

# bigger context window or different memory budget
python run_benchmark.py --max-model-len 8192 --gpu-memory-utilization 0.90
```

`run_benchmark.py --help` lists every flag.

## 4. Run the server manually (optional)

If you want to poke at the server independently of the orchestrator:

```bash
python launch_vllm.py --port 9000
```

Then in another shell:

```bash
curl http://127.0.0.1:9000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model": "llama3-toy-lora",
          "prompt": "Hello, world",
          "max_tokens": 16,
          "temperature": 0
        }'
```

Note the `model` field — it must match the adapter's logical name
(`--lora_name`, default `llama3-toy-lora`), not the base model path. That's
how vLLM routes the request through the adapter.

## 5. Output format

`output/timeline_results.csv` mirrors `examples/example_output.csv`:

| column | meaning |
| ------ | ------- |
| `idx` | original `row_id` in the timeline CSV (preserved across the run) |
| `t_rel_s` | seconds from timeline start when the request was actually fired |
| `latency_s` | wall-clock from request send to last token (or error) |
| `status` | `ok` on a 200, `http_<code>` or `error:<type>` otherwise |
| `ttft_s` | time from send to first non-empty streamed token |
| `avg_tbt_s` | mean inter-token gap for tokens after the first |
| `worst_tbt_s` | max inter-token gap for tokens after the first |

Warmup-phase requests are NOT recorded — same convention as
`auto_benchmark.py`.
