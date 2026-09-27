#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import time
import httpx
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"


async def measure_request(client, payload):
  t_submit = time.perf_counter()
  try:
    response = await client.post(
        API_URL, json={"texts": [payload]}, timeout=30.0
    )
    t_recv = time.perf_counter()

    if response.status_code != 200:
      print(f"⚠️ Server returned status {response.status_code}")
      return None, None, None

    total_latency = (t_recv - t_submit) * 1000.0  # ms

    # Безопасное извлечение заголовка
    header_val = response.headers.get("X-Process-Time-MS")
    if header_val is None:
      server_process_time = total_latency  # Fallback, если заголовок отсутствует
    else:
      server_process_time = float(header_val)

    queue_wait_time = max(0.0, total_latency - server_process_time)
    return total_latency, server_process_time, queue_wait_time

  except Exception as e:
    # Выводим единичные ошибки для диагностики
    # print(f"❌ Request error: {e}")
    return None, None, None


async def run_queue_profile(concurrency=100, total=1000):
  print(
      f"\n⚡ Profiling Queue vs Computation Latency (C={concurrency},"
      f" Requests={total})..."
  )
  payload = "Seni öldüreceğim seni aptal herif"

  # Увеличиваем таймаут пула соединений
  limits = httpx.Limits(
      max_connections=concurrency, max_keepalive_connections=concurrency
  )
  async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker():
      async with semaphore:
        return await measure_request(client, payload)

    results = await asyncio.gather(*[worker() for _ in range(total)])

  totals = [r[0] for r in results if r[0] is not None]
  cpu_times = [r[1] for r in results if r[1] is not None]
  queue_times = [r[2] for r in results if r[2] is not None]

  successful_requests = len(totals)
  print(
      f"📊 Execution finished: {successful_requests}/{total} successful"
      " requests."
  )

  if successful_requests == 0:
    print(
        "❌ Error: All requests failed! Ensure FastAPI is running on"
        " 127.0.0.1:8000."
    )
    return

  print("\n─── P99 DEGRADATION EMPIRICAL PROOF ───")
  print(f"📊 Total Latency P99:        {np.percentile(totals, 99):.2f} ms")
  print(
      "⚙️ Server CPU Inference P99:"
      f" {np.percentile(cpu_times, 99):.2f} ms"
  )
  print(
      "⏳ Queue / Event-Loop Delay P99:"
      f" {np.percentile(queue_times, 99):.2f} ms"
  )
  print(
      "💡 Queue Delay Overhead:     "
      f" {(np.percentile(queue_times, 99) / np.percentile(totals, 99)) * 100:.1f}%"
      " of total latency"
  )


if __name__ == "__main__":
  asyncio.run(run_queue_profile())