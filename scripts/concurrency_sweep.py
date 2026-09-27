#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refined High-Resolution Concurrency Sweep (C = 50..100, N=1000)
Includes Mathematically Valid Per-Request Queue Decomposition
Author: Maksym Nenashev (Systems Engineer)
"""

import asyncio
import time
import httpx
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"
CONCURRENCY_LEVELS = [50, 60, 70, 75, 80, 90, 100]
REQUESTS_PER_STEP = 1000
PAYLOAD = "Seni öldüreceğim seni aptal herif"

async def measure_request(client, semaphore):
    async with semaphore:
        t_submit = time.perf_counter()
        try:
            response = await client.post(API_URL, json={"texts": [PAYLOAD]}, timeout=60.0)
            t_recv = time.perf_counter()
            
            if response.status_code != 200:
                return None
            
            total_latency = (t_recv - t_submit) * 1000.0 # ms
            header_val = response.headers.get("X-Process-Time-MS")
            cpu_time = float(header_val) if header_val else total_latency
            queue_time = max(0.0, total_latency - cpu_time)
            
            # Позапросная декомпозиция (Per-request ratio)
            queue_ratio = queue_time / total_latency if total_latency > 0 else 0.0
            
            return {
                "total": total_latency,
                "cpu": cpu_time,
                "queue": queue_time,
                "queue_ratio": queue_ratio
            }
        except Exception:
            return None

async def run_step(concurrency, total_requests):
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    
    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        tasks = [measure_request(client, semaphore) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return None

    totals = [r["total"] for r in valid_results]
    cpus = [r["cpu"] for r in valid_results]
    queues = [r["queue"] for r in valid_results]
    ratios = [r["queue_ratio"] for r in valid_results]

    return {
        "c": concurrency,
        "n": len(valid_results),
        "p50_total": np.percentile(totals, 50),
        "p95_total": np.percentile(totals, 95),
        "p99_total": np.percentile(totals, 99),
        "p99_cpu": np.percentile(cpus, 99),
        "p99_queue": np.percentile(queues, 99),
        "p50_queue_ratio": np.percentile(ratios, 50) * 100.0, # Медианная доля очереди per request
        "mean_queue_ratio": np.mean(ratios) * 100.0           # Средняя доля очереди per request
    }

async def main():
    print("==================================================================")
    print("🔬 HIGH-RESOLUTION CONCURRENCY SWEEP (C = 50, 60, 70, 75, 80, 90, 100)")
    print("==================================================================")

    sweep_data = []
    for c in CONCURRENCY_LEVELS:
        print(f"🔄 Executing C={c} (N={REQUESTS_PER_STEP})...", end="", flush=True)
        res = await run_step(c, REQUESTS_PER_STEP)
        if res:
            sweep_data.append(res)
            print(f" ✅ Done ({res['n']}/{REQUESTS_PER_STEP} ok).")
        else:
            print(" ❌ Failed.")

    print("\n--- REFINED CONCURRENCY SWEEP TABLE FOR PAPER ---")
    print("| C | N | Total P50 (ms) | Total P95 (ms) | Total P99 (ms) | Inference P99 (ms) | Queue P99 (ms) | Mean Queue Share (%) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in sweep_data:
        print(f"| **{r['c']}** | {r['n']} | {r['p50_total']:.2f} | {r['p95_total']:.2f} | {r['p99_total']:.2f} | {r['p99_cpu']:.2f} | {r['p99_queue']:.2f} | **{r['mean_queue_ratio']:.1f}%** |")

if __name__ == "__main__":
    asyncio.run(main())