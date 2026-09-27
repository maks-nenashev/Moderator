#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Master Experiment Reconciliation Pipeline (Deep Recursive Scanner)
Path: scripts/reconcile_experiments_pipeline.py
Author: Maksym Nenashev (Systems Engineer)
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MANIFEST_FILE = ARTIFACTS_DIR / "PAPER_ARTIFACT_MANIFEST.json"

EXPECTED_ENGINES = [
    "v1", "v3", "v3.1", "v3.2", "v4", "v4.1", "v4.2",
    "v5", "v5.1", "v5.2", "v6", "v6.1", "v6.2",
    "v7", "v7.1", "v7.2", "v8", "v8.1", "v8.2", "v9", "v9.1", "v9.2"
]

REQUIRED_ARTIFACTS = {
    "dataset_protocol": ARTIFACTS_DIR / "dataset_protocol_summary.json",
    "rq1_proofs": ARTIFACTS_DIR / "statistical_proofs.json",
    "rq2_mcnemar": ARTIFACTS_DIR / "rq2_mcnemar_results.json",
    "rq4_concurrency": ARTIFACTS_DIR / "concurrency_sweep_refined.json",
    "rq4_batching": ARTIFACTS_DIR / "batch_scaling_results.json",
    "rq5_resilience": ARTIFACTS_DIR / "rq5_resilience_report.json"
}


def load_json_artifact(file_path: Path) -> Tuple[bool, Any]:
    if not file_path.exists():
        return False, None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return True, json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path.name}: {e}")
        return False, None


def extract_engines_from_rq1_deep(data: Any) -> set:
    """Рекурсивный сканер JSON-дерева для поиска движков из EXPECTED_ENGINES."""
    found = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k) in EXPECTED_ENGINES:
                    found.add(str(k))
                if isinstance(v, str) and v in EXPECTED_ENGINES:
                    found.add(v)
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str) and item in EXPECTED_ENGINES:
                    found.add(item)
                else:
                    walk(item)

    walk(data)
    return found


def validate_artifacts() -> Tuple[Dict[str, Any], List[str]]:
    loaded_data = {}
    errors = []

    print("==================================================================")
    print("🔄 EXECUTING RECONCILIATION & ARTIFACT LINEAGE VERIFICATION")
    print("==================================================================")

    # 1. Проверка наличия всех файлов артефактов
    for key, path in REQUIRED_ARTIFACTS.items():
        exists, data = load_json_artifact(path)
        if not exists:
            errors.append(f"Missing required artifact: {path.name}")
            print(f"❌ [FAIL] Artifact missing: {path.relative_to(BASE_DIR)}")
        else:
            loaded_data[key] = data
            print(f"✅ [PASS] Found artifact: {path.relative_to(BASE_DIR)}")

    if errors:
        return loaded_data, errors

    # 2. Глубокое извлечение 22 движков в RQ1
    rq1_raw = loaded_data.get("rq1_proofs", {})
    rq1_engines = extract_engines_from_rq1_deep(rq1_raw)

    missing_in_rq1 = set(EXPECTED_ENGINES) - rq1_engines
    if missing_in_rq1:
        errors.append(f"RQ1 incomplete. Missing engines ({len(missing_in_rq1)}/22): {sorted(list(missing_in_rq1))}")

    # 3. Проверка фазового перехода в RQ4 (Concurrency Sweep)
    rq4_sweep = loaded_data.get("rq4_concurrency", [])
    if isinstance(rq4_sweep, list) and len(rq4_sweep) > 0:
        c50_node = next((x for x in rq4_sweep if x.get("c") == 50), None)
        c60_node = next((x for x in rq4_sweep if x.get("c") == 60), None)
        if c50_node and c60_node:
            p99_50 = c50_node.get("p99_total", 0)
            p99_60 = c60_node.get("p99_total", 0)
            if p99_60 <= p99_50:
                errors.append(
                    f"RQ4 anomaly: P99 latency did not increase at C=60 transition "
                    f"({p99_50:.2f}ms vs {p99_60:.2f}ms)"
                )

    # 4. Проверка устойчивости в RQ5 (Resilience)
    rq5_data = loaded_data.get("rq5_resilience", {})
    scen_b = rq5_data.get("scenario_b", {})
    if scen_b and scen_b.get("recall", 0) <= 0:
        errors.append("RQ5 anomaly: Scenario B (Base Fallback) recall is 0.0")

    return loaded_data, errors


def main():
    loaded_data, errors = validate_artifacts()

    print("\n--- RECONCILIATION AUDIT SUMMARY ---")
    if errors:
        print(f"❌ STATUS: RECONCILIATION FAILED ({len(errors)} issues found)")
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)

    print("✅ STATUS: ALL EXPERIMENTAL ARTIFACTS VERIFIED & RECONCILED")

    manifest = {
        "reconciliation_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": sys.version.split()[0],
            "verified_engines_count": len(EXPECTED_ENGINES),
            "status": "VALIDATED"
        },
        "dataset_protocol": loaded_data.get("dataset_protocol"),
        "rq1_classification_quality": loaded_data.get("rq1_proofs"),
        "rq2_paired_specialization": loaded_data.get("rq2_mcnemar"),
        "rq4_scalability_concurrency": loaded_data.get("rq4_concurrency"),
        "rq4_scalability_batching": loaded_data.get("rq4_batching"),
        "rq5_resilience_degradation": loaded_data.get("rq5_resilience")
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Master Manifest generated successfully: {MANIFEST_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()