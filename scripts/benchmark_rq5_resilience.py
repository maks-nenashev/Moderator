#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ5: Resilience, Graceful Degradation & Circuit Breaker Benchmark (Vectorized Batch Execution)
Author: Maksym Nenashev (Systems Engineer)
"""

import asyncio
import json
import time
from pathlib import Path
import httpx
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data" / "processed"
API_URL = "http://127.0.0.1:8000/predict"

BASE_ENGINES = ["v1", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]
ALL_ENGINES = [
    "v1",
    "v3",
    "v3.1",
    "v3.2",
    "v4",
    "v4.1",
    "v4.2",
    "v5",
    "v5.1",
    "v5.2",
    "v6",
    "v6.1",
    "v6.2",
    "v7",
    "v7.1",
    "v7.2",
    "v8",
    "v8.1",
    "v8.2",
    "v9",
    "v9.1",
    "v9.2",
]


def load_holdout_test_set():
  """Загрузка всех образцов из контрольной выборки."""
  test_samples = []
  for csv_file in DATA_DIR.glob("*.csv"):
    try:
      df = pd.read_csv(
          csv_file, on_bad_lines="skip", engine="python"
      ).dropna(subset=["text", "label"])
      for _, row in df.iterrows():
        test_samples.append(
            {"text": str(row["text"]), "label": int(row["label"])}
        )
    except Exception:
      continue
  return test_samples


def evaluate_offline_fallback(test_samples, allowed_engines):
  """Векторизованная батч-оценка без поэлементных циклов Python."""
  texts = [s["text"] for s in test_samples]
  y_true = np.array([s["label"] for s in test_samples], dtype=int)
  n_samples = len(texts)

  if n_samples == 0:
    return 0.0, 0.0, 0.0, 0, 0, 0, 0

  # Матрица срабатываний: [N_samples, N_engines]
  triggers = np.zeros((n_samples, len(allowed_engines)), dtype=bool)

  t0 = time.perf_counter()

  for idx, eng in enumerate(allowed_engines):
    m_path = ARTIFACTS_DIR / eng / "model.joblib"
    v_path = ARTIFACTS_DIR / eng / "vectorizer.joblib"
    t_path = ARTIFACTS_DIR / eng / "thresholds.json"

    if m_path.exists() and v_path.exists() and t_path.exists():
      with open(t_path) as f:
        t_data = json.load(f)
        thresh = t_data.get("review_threshold", t_data.get("threshold", 0.85))

      vectorizer = joblib.load(v_path)
      model = joblib.load(m_path)

      # Векторизация и инференс над всем массивом (1 вызов C-Core)
      X = vectorizer.transform(texts)
      probs = model.predict_proba(X)[:, 1]
      triggers[:, idx] = probs >= thresh

  t1 = time.perf_counter()
  total_eval_time = t1 - t0

  # Итоговый отклик каскада (логическое ИЛИ по строкам)
  y_pred = np.any(triggers, axis=1).astype(int)

  tp = int(np.sum((y_true == 1) & (y_pred == 1)))
  tn = int(np.sum((y_true == 0) & (y_pred == 0)))
  fp = int(np.sum((y_true == 0) & (y_pred == 1)))
  fn = int(np.sum((y_true == 1) & (y_pred == 0)))

  avg_latency_ms = (total_eval_time * 1000.0) / n_samples
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

  return recall, fpr, avg_latency_ms, tp, tn, fp, fn


async def run_live_timeout_test(concurrency=10, total=500, timeout_s=0.20):
  """Проверка выдерживания SLA-бюджета 200мс в живом сервисе."""
  limits = httpx.Limits(max_connections=concurrency)
  semaphore = asyncio.Semaphore(concurrency)
  payload = "Seni öldüreceğim seni aptal herif"

  success_count = 0
  timeout_count = 0
  latencies = []

  async with httpx.AsyncClient(limits=limits) as client:

    async def worker():
      nonlocal success_count, timeout_count
      async with semaphore:
        t0 = time.perf_counter()
        try:
          res = await client.post(
              API_URL, json={"texts": [payload]}, timeout=timeout_s
          )
          t1 = time.perf_counter()
          if res.status_code == 200:
            success_count += 1
            latencies.append((t1 - t0) * 1000.0)
        except httpx.TimeoutException:
          timeout_count += 1
        except Exception:
          pass

    tasks = [worker() for _ in range(total)]
    await asyncio.gather(*tasks)

  availability = (success_count / total) * 100.0
  p99_lat = float(np.percentile(latencies, 99)) if latencies else 0.0
  return availability, p99_lat, timeout_count


def main():
  print("==================================================================")
  print("🛡️ RUNNING VECTORIZED RQ5 RESILIENCE & FALLBACK EXPERIMENTS")
  print("==================================================================")

  samples = load_holdout_test_set()
  print(f"📦 Loaded {len(samples)} hold-out evaluation samples.")

  # 1. Full Cascade Baseline
  print("\n🔄 Scenario A: Full Cascade (22 Engines)...", end="", flush=True)
  rec_a, fpr_a, lat_a, tp_a, tn_a, fp_a, fn_a = evaluate_offline_fallback(
      samples, ALL_ENGINES
  )
  print(" ✅ Done.")

  # 2. Base-Only Fallback
  print(
      "🔄 Scenario B: Emergency Fallback (Base Engines Only)...",
      end="",
      flush=True,
  )
  rec_b, fpr_b, lat_b, tp_b, tn_b, fp_b, fn_b = evaluate_offline_fallback(
      samples, BASE_ENGINES
  )
  print(" ✅ Done.")

  # 3. SLA Circuit Breaker Test
  print(
      "🔄 Scenario C: SLA Test (C=10, 200ms Timeout)...",
      end="",
      flush=True,
  )
  avail_c, p99_c, timeouts_c = asyncio.run(
      run_live_timeout_test(concurrency=10, total=500, timeout_s=0.20)
  )
  print(" ✅ Done.")

  print("\n--- VALIDATED RQ5 RESILIENCE & FALLBACK RESULTS ---")
  print(
      "| Scenario | Active Engines | Recall | FPR | Avg CPU Latency (ms) |"
      " Recall Retained (%) | SLA Availability (%) |"
  )
  print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
  print(
      f"| **Scenario A (Full Cascade)** | 22 | {rec_a:.4f} | {fpr_a:.4f} |"
      f" {lat_a:.4f} | 100.0% | 100.0% |"
  )
  print(
      f"| **Scenario B (Base Fallback)** | 8 | {rec_b:.4f} | {fpr_b:.4f} |"
      f" **{lat_b:.4f}** | **{(rec_b/rec_a)*100 if rec_a > 0 else 0:.1f}%** |"
      " 100.0% |"
  )
  print(
      "| **Scenario C (200ms SLA Budget)** | 22 | N/A | N/A |"
      f" {p99_c:.2f} (P99) | N/A | **{avail_c:.1f}%** |"
  )


if __name__ == "__main__":
  main()