"""
launch_vllm.py — start a vLLM 0.18.0 OpenAI-compatible server with one LoRA
adapter loaded.

The orchestrator (run_benchmark.py) shells out to this script with
`python launch_vllm.py ...`, exactly the same pattern as
examples/launch_llama3.py. Keep the surface minimal — only the knobs the
orchestrator actually passes through.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = "/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
DEFAULT_LORA_DIR = SCRIPT_DIR / "llama3-toy-lora-ft"
DEFAULT_LORA_NAME = "llama3-toy-lora"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Base model path (local dir or HF id).")
    ap.add_argument("--lora_dir", default=str(DEFAULT_LORA_DIR),
                    help="LoRA adapter directory (contains adapter_config.json).")
    ap.add_argument("--lora_name", default=DEFAULT_LORA_NAME,
                    help="Logical name the client uses in the `model` field.")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--max-loras", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    lora_modules = f"{args.lora_name}={args.lora_dir}"

    # Fairness flags vs DeltaServe. The benchmark's prompts share a long
    # deterministic prefix ("Instruction:\nhello hello hello..."), so vLLM's
    # default prefix caching gives it near-free prefill on most of every
    # request after the first — DeltaServe has no prefix cache, so it would
    # be doing strictly more work per request. Disable it to isolate raw
    # engine speed from "feature DeltaServe doesn't have."
    parts = [
        "vllm", "serve", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--enable-lora",
        "--lora-modules", lora_modules,
        "--max-lora-rank", str(args.max_lora_rank),
        "--max-loras", str(args.max_loras),
        "--max-cpu-loras", str(args.max_loras),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
        "--no-enable-prefix-caching",
    ]

    cmd = " ".join(shlex.quote(p) for p in parts)
    print(cmd, flush=True)
    os.execvp("bash", ["bash", "-c", cmd])


if __name__ == "__main__":
    main()
