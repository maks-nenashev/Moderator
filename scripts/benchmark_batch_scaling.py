#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Size Scaling Profiler (Fixed C=50, N=1000)
Path: scripts/benchmark_batch_scaling.py
"""

import asyncio
import json
import time
from pathlib import Path
import httpx
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
API_URL = "http://127.0.0.1:8000/predict"
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
CONCURRENCY = 50
TARGET_SAMPLE_COUNT = 1000
PAYLOAD_TEXT = "Seni öldüreceğim seni aptal herif"


async def send_batch(client, semaphore, batch_size):
  async with semaphore:
    payload = {"texts": [PAYLOAD_TEXT] * batch_size}
    t_sub = time.perf_counter()
    try:
      res = await client.post(API_URL, json=payload, timeout=60.0)
      t_rec = time.perf_counter()
      if res.status_code != 200:
        return None
      total_ms = (t_rec - t_sub) * 1000.0
      cpu_ms = float(res.headers.get("X-Process-Time-MS", total_ms))
      queue_ms = max(0.0, total_ms - cpu_ms)
      return total_ms, cpu_ms, queue_ms
    except Exception:
      return None


async def run_batch_step(batch_size):
  n_requests = TARGET_SAMPLE_COUNT // batch_size
  limits = httpx.Limits(
      max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY
  )
  semaphore = asyncio.Semaphore(CONCURRENCY)

  t_start = time.perf_counter()
  async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
    tasks = [
        send_batch(client, semaphore, batch_size) for _ in range(n_requests)
    ]
    results = await asyncio.gather(*tasks)
  t_elapsed = time.perf_counter() - t_start

  valid = [r for r in results if r is not None]
  if not valid:
    return None

  totals = [r[0] for r in valid]
  cpus = [r[1] for r in valid]
  queues = [r[2] for r in valid]
  rps = (len(valid) * batch_size) / t_elapsed

  return {
      "b": batch_size,
      "n_req": len(valid),
      "rps": round(float(rps), 1),
      "p50_total": round(float(np.percentile(totals, 50)), 2),
      "p95_total": round(float(np.percentile(totals, 95)), 2),
      "p99_total": round(float(np.percentile(totals, 99)), 2),
      "p99_cpu": round(float(np.percentile(cpus, 99)), 2),
      "p99_queue": round(float(np.percentile(queues, 99)), 2),
  }


async def main():
  print("==================================================================")
  print(
      f"⚡ RUNNING RQ4 BATCH SCALING SWEEP (Fixed C={CONCURRENCY},"
      " Samples ≈ 1000)"
  )
  print("==================================================================")

  out = []
  for b in BATCH_SIZES:
    print(f"🔄 Testing Batch Size B={b}...", end="", flush=True)
    res = await run_batch_step(b)
    if res:
      out.append(res)
      print(" ✅ Done.")

  # Запись артефакта
  out_path = ARTIFACTS_DIR / "batch_scaling_results.json"
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
  print(f"\n✅ Saved artifact to {out_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
  asyncio.run(main())