#!/usr/bin/env python3
"""
run_benchmark.py — fire a timeline of LoRA-inference requests at a vLLM
0.18.0 OpenAI-compatible server and record per-request metrics.

Mirrors examples/auto_benchmark.py:
  - launches launch_vllm.py in a separate process group (POSIX only)
  - streams server logs live with a [server] prefix
  - waits for /health
  - warmup phase: replays the first N rows on their timeline schedule,
    NOT recorded
  - timeline phase: replays ALL rows on their timeline schedule
  - writes per-request CSV with columns:
        idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s
    (same shape as examples/example_output.csv)

Per-request streaming: each request uses the OpenAI /v1/completions
endpoint with stream=true. We measure:
  ttft_s     — wall-clock from request send to first non-empty token
  avg_tbt_s  — mean inter-token gap for tokens after the first
  worst_tbt_s — max inter-token gap for tokens after the first
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

# Active progress bar — log pump routes output through tqdm.write so the
# bar stays anchored at the bottom while the timeline runs.
_active_pbar: Optional["tqdm"] = None


def _emit_line(line: str) -> None:
    if _active_pbar is not None:
        tqdm.write(line, file=sys.stdout)
    else:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def make_prompt_from_length(prompt_length: int) -> str:
    """Deterministic filler prompt approximately matching `prompt_length`
    characters. Same recipe as examples/auto_benchmark.py so the two
    benchmarks are comparable on prompt shape."""
    base = "Instruction:\n"
    tail = "\n### Response: "
    filler_needed = max(0, prompt_length - len(base) - len(tail))
    filler = ("hello " * 10000)[:filler_needed]
    prompt = base + filler + tail
    if len(prompt) < prompt_length:
        prompt += "x" * (prompt_length - len(prompt))
    return prompt[:prompt_length]


@dataclass
class TimelineRow:
    timestamp_s: float
    prompt_length: int
    max_new_tokens: int
    row_id: int


def load_timeline_csv(path: str) -> List[TimelineRow]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Timeline CSV not found: {path}")
    rows: List[TimelineRow] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp_s", "prompt_length", "max_new_tokens"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Timeline CSV missing required columns: {sorted(missing)}")
        for i, r in enumerate(reader):
            rows.append(
                TimelineRow(
                    timestamp_s=float(r["timestamp_s"]),
                    prompt_length=int(float(r["prompt_length"])),
                    max_new_tokens=int(float(r["max_new_tokens"])),
                    row_id=i,
                )
            )
    rows.sort(key=lambda x: x.timestamp_s)
    return rows


def write_results_csv(
    path: str,
    rows: List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_rel_s", "latency_s", "status", "ttft_s", "avg_tbt_s", "worst_tbt_s"])
        for idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt in rows:
            w.writerow([idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt])
    print(f"[orchestrator] Wrote results CSV: {path}", flush=True)


# ----------------------------
# HTTP helpers
# ----------------------------
async def try_health(session: aiohttp.ClientSession, server: str, timeout_s: float = 1.0) -> bool:
    try:
        async with session.get(f"{server.rstrip('/')}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


async def wait_for_server(
    server: str,
    max_wait_s: float = 600.0,
    poll_period_s: float = 1.0,
) -> None:
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while True:
            if await try_health(session, server):
                print(f"[orchestrator] Server is up at {server}", flush=True)
                return
            if time.monotonic() - t0 > max_wait_s:
                raise TimeoutError(f"Server didn't become healthy within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_period_s)


async def send_one_request(
    session: aiohttp.ClientSession,
    server: str,
    idx: int,
    t_rel: float,
    prompt: str,
    model_name: str,
    max_new_tokens: int,
) -> Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]:
    """
    Streaming OpenAI /v1/completions. Computes ttft from first token,
    avg_tbt/worst_tbt from inter-token gaps after the first.
    """
    url = f"{server.rstrip('/')}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }

    ttft: Optional[float] = None
    tbts: List[float] = []
    last_tok_t: Optional[float] = None

    t_send = time.monotonic()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                latency = time.monotonic() - t_send
                return (idx, t_rel, latency, f"http_{resp.status}", None, None, None)

            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if not text:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t_send
                    last_tok_t = now
                else:
                    if last_tok_t is not None:
                        tbts.append(now - last_tok_t)
                    last_tok_t = now

        latency = time.monotonic() - t_send
        avg_tbt = (sum(tbts) / len(tbts)) if tbts else None
        worst_tbt = max(tbts) if tbts else None
        return (idx, t_rel, latency, "ok", ttft, avg_tbt, worst_tbt)
    except Exception as e:
        latency = time.monotonic() - t_send
        return (idx, t_rel, latency, f"error:{type(e).__name__}", None, None, None)


# ----------------------------
# Scheduling
# ----------------------------
WARMUP_START_OFFSET_S = 1.0


async def run_warmup_requests(
    server: str,
    model_name: str,
    warmup_rows: List[TimelineRow],
    stop_event: asyncio.Event,
    request_timeout_s: float = 600.0,
) -> None:
    if not warmup_rows:
        return
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    base_ts = min(r.timestamp_s for r in warmup_rows)
    span = max(r.timestamp_s for r in warmup_rows) - base_ts

    async with aiohttp.ClientSession(
        headers={"User-Agent": "WarmupClient"},
        connector=connector,
        timeout=timeout,
    ) as session:
        t0 = time.monotonic()
        print(
            f"[orchestrator] Warmup: {len(warmup_rows)} requests following timeline "
            f"(first at +{WARMUP_START_OFFSET_S:.2f}s, span {span:.2f}s)",
            flush=True,
        )

        async def _run_one(row: TimelineRow) -> None:
            target_rel = (row.timestamp_s - base_ts) + WARMUP_START_OFFSET_S
            target_abs = t0 + target_rel
            delay = target_abs - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop_event.is_set():
                return
            t_rel = time.monotonic() - t0
            prompt = make_prompt_from_length(row.prompt_length)
            idx, _, latency, status, ttft, _, _ = await send_one_request(
                session=session, server=server,
                idx=row.row_id, t_rel=t_rel, prompt=prompt,
                model_name=model_name, max_new_tokens=row.max_new_tokens,
            )
            if idx % 10 == 0:
                print(
                    f"[warmup] idx={idx} t_rel={t_rel:.3f}s latency={latency:.3f}s "
                    f"status={status} ttft={ttft}", flush=True,
                )

        tasks = [asyncio.create_task(_run_one(row)) for row in warmup_rows]
        await asyncio.gather(*tasks)
        print("[orchestrator] Warmup completed", flush=True)


async def run_timeline_requests(
    server: str,
    model_name: str,
    timeline_rows: List[TimelineRow],
    stop_event: asyncio.Event,
    request_timeout_s: float = 600.0,
) -> List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]]:
    if not timeline_rows:
        print("[orchestrator] No timeline rows to run.", flush=True)
        return []

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    results: List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]] = []

    base_ts = min(r.timestamp_s for r in timeline_rows)
    total_duration = max(r.timestamp_s for r in timeline_rows) - base_ts

    async with aiohttp.ClientSession(
        headers={"User-Agent": "TimelineClient"},
        connector=connector,
        timeout=timeout,
    ) as session:
        t0 = time.monotonic()
        print(
            f"[orchestrator] Timeline start (base_ts={base_ts:.6f}), "
            f"{len(timeline_rows)} requests, total schedule {total_duration:.2f}s",
            flush=True,
        )

        global _active_pbar
        pbar = tqdm(
            total=max(total_duration, 1e-3),
            desc="Timeline",
            unit="s",
            bar_format="{desc}: {percentage:5.1f}%|{bar}| {n:6.1f}/{total:.1f}s "
                       "[{elapsed}<{remaining}]",
            leave=True,
            position=0,
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.1,
        )
        _active_pbar = pbar

        async def _progress_updater() -> None:
            last = 0.0
            while not stop_event.is_set():
                elapsed = min(time.monotonic() - t0, total_duration)
                delta = elapsed - last
                if delta > 0:
                    pbar.update(delta)
                    last = elapsed
                if elapsed >= total_duration:
                    return
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                    return
                except asyncio.TimeoutError:
                    continue

        progress_task = asyncio.create_task(_progress_updater())

        async def _run_one(row: TimelineRow) -> None:
            target_rel = row.timestamp_s - base_ts
            target_abs = t0 + target_rel
            delay = target_abs - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop_event.is_set():
                return
            t_rel = time.monotonic() - t0
            prompt = make_prompt_from_length(row.prompt_length)
            result = await send_one_request(
                session=session, server=server,
                idx=row.row_id, t_rel=t_rel, prompt=prompt,
                model_name=model_name, max_new_tokens=row.max_new_tokens,
            )
            results.append(result)

        tasks = [asyncio.create_task(_run_one(row)) for row in timeline_rows]
        try:
            await asyncio.gather(*tasks)
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except (asyncio.CancelledError, Exception):
                pass
            if not stop_event.is_set() and pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            pbar.refresh()
            pbar.close()
            _active_pbar = None

    results.sort(key=lambda x: x[0])
    print("[orchestrator] Timeline completed", flush=True)
    return results


# ----------------------------
# Process orchestration (POSIX only)
# ----------------------------
def terminate_process_tree_fast(p: subprocess.Popen, grace_s: float = 0.5) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline_csv",
                    default=str(SCRIPT_DIR / "timeline.csv"),
                    help="Path to the request schedule CSV.")
    ap.add_argument("--model",
                    default="/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                    help="Base model dir (passed to vllm serve).")
    ap.add_argument("--lora_dir",
                    default=str(SCRIPT_DIR / "llama3-toy-lora-ft"),
                    help="LoRA adapter dir.")
    ap.add_argument("--lora_name", default="llama3-toy-lora",
                    help="Adapter name; this is what the client sends in `model`.")
    ap.add_argument("--launcher", default=str(SCRIPT_DIR / "launch_vllm.py"))
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out_csv", default=str(SCRIPT_DIR / "output" / "timeline_results.csv"))
    ap.add_argument("--warmup_count", type=int, default=20,
                    help="Cap on number of warmup rows taken from the start of the timeline.")
    ap.add_argument("--warmup_duration_s", type=float, default=10.0,
                    help="Cap on warmup window in seconds (timeline-time).")
    ap.add_argument("--warmup_rest_s", type=float, default=2.0,
                    help="Rest after warmup before firing the full timeline.")
    ap.add_argument("--server_max_wait_s", type=float, default=600.0,
                    help="How long to wait for /health after launch.")
    args = ap.parse_args()

    timeline_rows = load_timeline_csv(args.timeline_csv)
    print(f"[orchestrator] Loaded {len(timeline_rows)} rows from {args.timeline_csv}", flush=True)
    if timeline_rows:
        print(
            f"[orchestrator] Timeline range: {timeline_rows[0].timestamp_s:.6f}s "
            f"-> {timeline_rows[-1].timestamp_s:.6f}s",
            flush=True,
        )

    server = f"http://127.0.0.1:{args.port}"

    cmd = [
        sys.executable, "-u", args.launcher,
        "--model", args.model,
        "--lora_dir", args.lora_dir,
        "--lora_name", args.lora_name,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    print("[orchestrator] launching:", " ".join(cmd), flush=True)
    print(f"[orchestrator] writing results CSV to: {args.out_csv}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    p: Optional[subprocess.Popen] = None
    log_task: Optional[asyncio.Task] = None
    stop_event = asyncio.Event()

    try:
        # Raw bytes so '\r' from tqdm progress bars survives universal-newline
        # translation; the pumper splits on '\n' and '\r' separately.
        p = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        async def pump_logs() -> None:
            assert p is not None and p.stdout is not None
            loop = asyncio.get_running_loop()
            stream = p.stdout
            buf: List[str] = []
            redrawing = False

            def read_chunk() -> bytes:
                return stream.read(256)

            while True:
                raw = await loop.run_in_executor(None, read_chunk)
                if not raw:
                    if buf:
                        _emit_line(("\r" if redrawing else "") + "[server] " + "".join(buf))
                    break
                chunk = raw.decode("utf-8", errors="replace")
                for ch in chunk:
                    if ch == "\n":
                        prefix = "\r" if redrawing else ""
                        _emit_line(prefix + "[server] " + "".join(buf))
                        buf.clear()
                        redrawing = False
                    elif ch == "\r":
                        if _active_pbar is not None:
                            _emit_line("[server] " + "".join(buf))
                        else:
                            sys.stdout.write("\r[server] " + "".join(buf))
                            sys.stdout.flush()
                        buf.clear()
                        redrawing = _active_pbar is None
                    else:
                        buf.append(ch)

        log_task = asyncio.create_task(pump_logs())

        loop = asyncio.get_running_loop()

        def _on_sigint() -> None:
            if p is not None and p.poll() is None:
                terminate_process_tree_fast(p, grace_s=0.5)
            stop_event.set()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        waiter = asyncio.create_task(
            wait_for_server(server=server, max_wait_s=args.server_max_wait_s, poll_period_s=1.0)
        )
        stopper = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if stop_event.is_set():
            return

        # Warmup row selection — same logic as auto_benchmark.py:
        # take min(by_count, by_duration) rows from the head of the timeline.
        if timeline_rows and args.warmup_count > 0 and args.warmup_duration_s > 0:
            base_ts = timeline_rows[0].timestamp_s
            by_count = min(args.warmup_count, len(timeline_rows))
            by_duration = 0
            for r in timeline_rows:
                if r.timestamp_s - base_ts <= args.warmup_duration_s:
                    by_duration += 1
                else:
                    break
            warmup_n = min(by_count, by_duration)
        else:
            warmup_n = 0
        warmup_rows = timeline_rows[:warmup_n]
        print(
            f"[orchestrator] Warmup row selection: count_cap={args.warmup_count}, "
            f"duration_cap={args.warmup_duration_s:.2f}s -> {len(warmup_rows)} rows",
            flush=True,
        )

        if warmup_rows:
            await run_warmup_requests(
                server=server, model_name=args.lora_name,
                warmup_rows=warmup_rows, stop_event=stop_event,
            )
            if stop_event.is_set():
                return
            if args.warmup_rest_s > 0:
                print(f"[orchestrator] Resting {args.warmup_rest_s:.2f}s after warmup...", flush=True)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=args.warmup_rest_s)
                    return
                except asyncio.TimeoutError:
                    pass

        results = await run_timeline_requests(
            server=server, model_name=args.lora_name,
            timeline_rows=timeline_rows, stop_event=stop_event,
        )
        write_results_csv(args.out_csv, results)

    except KeyboardInterrupt:
        if p is not None:
            terminate_process_tree_fast(p, grace_s=0.5)
    finally:
        if p is not None and p.poll() is None:
            print("[orchestrator] shutting down server...", flush=True)
            terminate_process_tree_fast(p, grace_s=1.0)
        if log_task is not None:
            log_task.cancel()
            try:
                await asyncio.wait_for(log_task, timeout=0.5)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
