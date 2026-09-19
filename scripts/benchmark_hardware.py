#!/usr/bin/env python3
# -*- coding: utf-8 -*-
                                        # python3 scripts/benchmark_hardware.py

"""
Hardware & Performance Benchmarking Utility for Data Sentinel NLP
Author: Maksym Nenashev (Systems Engineer)
"""

import asyncio
import time
import json
import psutil
import httpx
import numpy as np
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"
METRICS_URL = "http://127.0.0.1:8000/metrics"

# Набор тестовых фраз для загрузки каскадов всех 7 регионов
TEST_PAYLOADS = [
    "Säljer en begagnad cykel i gott skick",
    "Jag ska döda dig din jävla idiot",
    "v.i.t.u.n_k.u.s.i.p.a.a",
    "Snabba pengar utan avtal och dokument, skriv till PM",
    "Prodajem rabljeni bicikl u dobrom stanju",
    "Ubit ću te ti jebeni idiote",
    "j_e_b_e_n_i_i_d_i_o_t",
    "Brzi novac bez ugovora i dokumenata javite se u DM",
    "İyi durumda satılık ikinci el bisiklet",
    "Seni öldüreceğim seni aptal herif",
    "a_p_t_a_l_h_e_r_i_f",
    "Sözleşmesiz ve belgesiz hızlı para DMden yazın"
]

def get_process_memory():
    """Возвращает потребление RAM текущей базовой системы и дочерних процессов FastAPI."""
    current_process = psutil.Process()
    mem_info = current_process.memory_info().rss / (1024 * 1024) # MB
    
    # Ищем процессы uvicorn/fastapi
    for child in current_process.children(recursive=True):
        mem_info += child.memory_info().rss / (1024 * 1024)
    return mem_info

async def send_batch(client: httpx.AsyncClient, batch: list):
    start = time.perf_counter()
    try:
        response = await client.post(API_URL, json={"texts": batch}, timeout=10.0)
        elapsed = (time.perf_counter() - start) * 1000.0 # ms
        return elapsed, response.status_code
    except Exception as e:
        return None, 500

async def run_benchmark(concurrency: int, total_requests: int, batch_size: int):
    print(f"\n⚡ Running Test: Concurrency={concurrency}, Total Requests={total_requests}, Batch Size={batch_size}")
    
    # Срез ресурсов ДО теста
    mem_before = get_process_memory()
    cpu_before = psutil.cpu_percent(interval=None)
    
    payload_batch = (TEST_PAYLOADS * ((batch_size // len(TEST_PAYLOADS)) + 1))[:batch_size]
    
    latencies = []
    status_codes = []
    
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def worker():
            async with semaphore:
                lat, code = await send_batch(client, payload_batch)
                if lat is not None:
                    latencies.append(lat)
                    status_codes.append(code)

        tasks = [worker() for _ in range(total_requests)]
        
        bench_start = time.perf_counter()
        await asyncio.gather(*tasks)
        bench_duration = time.perf_counter() - bench_start

    # Срез ресурсов ПОСЛЕ теста
    mem_after = get_process_memory()
    cpu_after = psutil.cpu_percent(interval=None)

    total_texts = total_requests * batch_size
    rps = total_texts / bench_duration if bench_duration > 0 else 0

    print("─── Results ───")
    print(f"⏱️ Total Execution Time: {bench_duration:.2f} s")
    print(f"🚀 Throughput:           {rps:.2f} texts/sec")
    print(f"📊 Latency P50 (Median): {np.percentile(latencies, 50):.2f} ms")
    print(f"📊 Latency P95:          {np.percentile(latencies, 95):.2f} ms")
    print(f"📊 Latency P99:          {np.percentile(latencies, 99):.2f} ms")
    print(f"🧠 RAM RSS Delta:        {mem_after - mem_before:+.2f} MB (Peak: {mem_after:.2f} MB)")
    print(f"💻 Avg CPU Usage:        {cpu_after:.1f}%")
    
    return {
        "concurrency": concurrency,
        "batch_size": batch_size,
        "total_texts": total_texts,
        "duration_sec": bench_duration,
        "rps": rps,
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "ram_rss_mb": mem_after
    }

async def main():
    print("==================================================")
    print("🔬 DATA SENTINEL NLP — HARDWARE & LATENCY PROFILER")
    print("==================================================")
    
    # Сценарии нагрузки для статьи
    scenarios = [
        {"concurrency": 1,   "total_requests": 200, "batch_size": 1},   # Sequential Latency
        {"concurrency": 10,  "total_requests": 500, "batch_size": 8},   # Moderate Load
        {"concurrency": 50,  "total_requests": 1000, "batch_size": 16}, # High Concurrency
        {"concurrency": 100, "total_requests": 2000, "batch_size": 32}  # Peak Stress Test
    ]
    
    results = []
    for sc in scenarios:
        res = await run_benchmark(sc["concurrency"], sc["total_requests"], sc["batch_size"])
        results.append(res)
        await asyncio.sleep(2) # Пауза для стабилизации CPU/RAM

    # Сохранение итогового отчета в JSON для построения графиков
    report_path = Path("artifacts/benchmark_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Benchmark report saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())