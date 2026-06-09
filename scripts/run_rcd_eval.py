"""
Batch-run RCD algorithm on RCAEval-compatible datasets.

Evaluation follows RCAEval's methodology (main.py):
- AC@k: fraction of cases where ground-truth is in top-k predictions
- Avg@k: average of AC@1 through AC@k
- Two levels: service-level (coarse) and metric-level (fine)
- Fault type normalization: delay→latency, loss→latency, disk→diskio
- Output format matches RCAEval: Avg@5-{FAULT_TYPE} per fault type

Dataset layout:
    <dataset_dir>/<service>_<fault>/<case_id>/data.csv
    <dataset_dir>/<service>_<fault>/<case_id>/inject_time.txt

Usage:
    python scripts/run_rcd_eval.py [--dataset-dir data/RCAEval/online-boutique] [--gamma 5]
"""

import argparse
import json
import logging
import sys
import time as _time
from pathlib import Path

import pandas as pd

# Suppress RCD internal logs
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logging.getLogger("app.tools.rcd.rcd").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.rcd_wrapper import run_rcd_analysis


# =========== Node and Evaluator (matching RCAEval) ===========

class Node:
    __slots__ = ("entity", "metric")

    def __init__(self, entity: str, metric: str):
        self.entity = entity
        self.metric = metric

    def __eq__(self, other):
        return isinstance(other, Node) and self.entity == other.entity and self.metric == other.metric

    def __hash__(self):
        return hash((self.entity, self.metric))

    def __repr__(self):
        return f"Node({self.entity}, {self.metric})"


class Evaluator:
    def __init__(self):
        self._accuracy = {k: 0.0 for k in range(1, 6)}
        self._accuracy_service = {k: 0.0 for k in range(1, 6)}
        self._ranks = []

    def add_case(self, ranks, answer: Node):
        self._ranks.append(ranks[:5])
        service_ranks = [n.entity for n in ranks]
        service_answer = answer.entity
        for k in range(1, 6):
            self._accuracy[k] += int(answer in ranks[:k])
            self._accuracy_service[k] += int(service_answer in service_ranks[:k])

    @property
    def num(self):
        return len(self._ranks)

    def accuracy(self, k):
        return self._accuracy[k] / self.num if self._ranks else None

    def accuracy_service(self, k):
        return self._accuracy_service[k] / self.num if self._ranks else None

    def average(self, k):
        return sum(self.accuracy(i) for i in range(1, k + 1)) / k if self._ranks else None

    def average_service(self, k):
        return sum(self.accuracy_service(i) for i in range(1, k + 1)) / k if self._ranks else None


# =========== Helpers ===========

def build_node_ranks(ranks_str):
    """Convert RCD string ranks to Node lists for both evaluation levels."""
    s_ranks = [Node(r.split("_")[0].replace("-db", ""), "unknown") for r in ranks_str]
    # Deduplicate service-level (keep first occurrence)
    seen = set()
    s_ranks_dedup = []
    for n in s_ranks:
        if n.entity not in seen:
            seen.add(n.entity)
            s_ranks_dedup.append(n)
    f_ranks = []
    for r in ranks_str:
        parts = r.split("_", 1)
        f_ranks.append(Node(parts[0], parts[1] if len(parts) > 1 else "unknown"))
    return s_ranks_dedup, f_ranks


def load_sample(sample_dir: Path):
    csv_path = sample_dir / "data.csv"
    inject_path = sample_dir / "inject_time.txt"
    df = pd.read_csv(csv_path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    inject_time = float(inject_path.read_text().strip())
    return df, inject_time


# =========== Main ===========

def run_eval(dataset_dir: Path, gamma: int):
    scenarios = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if not scenarios:
        print(f"No scenarios found in {dataset_dir}")
        return

    # Per fault-type evaluators (matching RCAEval main.py naming)
    s_evaluator_cpu = Evaluator();   f_evaluator_cpu = Evaluator()
    s_evaluator_mem = Evaluator();   f_evaluator_mem = Evaluator()
    s_evaluator_lat = Evaluator();   f_evaluator_lat = Evaluator()
    s_evaluator_loss = Evaluator();  f_evaluator_loss = Evaluator()
    s_evaluator_io = Evaluator();    f_evaluator_io = Evaluator()
    s_evaluator_socket = Evaluator();f_evaluator_socket = Evaluator()
    s_evaluator_all = Evaluator();   f_evaluator_all = Evaluator()

    total = 0
    success = 0
    total_elapsed = 0.0

    for scenario_dir in scenarios:
        scenario_name = scenario_dir.name  # e.g. "adservice_cpu"
        parts = scenario_name.split("_", 1)
        if len(parts) != 2:
            continue
        service, fault = parts

        sample_dirs = sorted(
            (d for d in scenario_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )

        for sample_dir in sample_dirs:
            total += 1
            print(f"[{total}] {scenario_name}/{sample_dir.name} ... ", end="", flush=True)

            output_dir = sample_dir / "rcd_result"
            output_dir.mkdir(exist_ok=True)

            try:
                df, inject_time = load_sample(sample_dir)
                t0 = _time.time()
                result = run_rcd_analysis(data=df, inject_time=inject_time, gamma=gamma)
                elapsed = _time.time() - t0
                total_elapsed += elapsed

                ranks = result.get("root_causes", [])
                s_ranks, f_ranks = build_node_ranks(ranks)

                s_answer = Node(service, "unknown")

                # Metric-level answer uses normalized fault name
                fault_map = {"delay": "latency", "loss": "latency", "disk": "diskio"}
                f_answer_metric = fault_map.get(fault, fault)

                f_answer = Node(service, f_answer_metric)

                # Route to fault-type evaluators (matching RCAEval main.py exactly)
                if fault == "cpu":
                    s_evaluator_cpu.add_case(s_ranks, s_answer)
                    f_evaluator_cpu.add_case(f_ranks, f_answer)
                    s_evaluator_all.add_case(s_ranks, s_answer)
                    f_evaluator_all.add_case(f_ranks, f_answer)
                elif fault == "mem":
                    s_evaluator_mem.add_case(s_ranks, s_answer)
                    f_evaluator_mem.add_case(f_ranks, f_answer)
                    s_evaluator_all.add_case(s_ranks, s_answer)
                    f_evaluator_all.add_case(f_ranks, f_answer)
                elif fault == "delay":
                    s_evaluator_lat.add_case(s_ranks, s_answer)
                    f_evaluator_lat.add_case(f_ranks, f_answer)
                    s_evaluator_all.add_case(s_ranks, s_answer)
                    f_evaluator_all.add_case(f_ranks, f_answer)
                elif fault == "loss":
                    s_evaluator_loss.add_case(s_ranks, s_answer)
                    f_evaluator_loss.add_case(f_ranks, f_answer)
                    s_evaluator_all.add_case(s_ranks, s_answer)
                    f_evaluator_all.add_case(f_ranks, f_answer)
                elif fault == "disk":
                    s_evaluator_io.add_case(s_ranks, s_answer)
                    f_evaluator_io.add_case(f_ranks, f_answer)
                    s_evaluator_all.add_case(s_ranks, s_answer)
                    f_evaluator_all.add_case(f_ranks, f_answer)
                elif fault == "socket":
                    s_evaluator_socket.add_case(s_ranks, s_answer)
                    f_evaluator_socket.add_case(f_ranks, f_answer)
                    s_evaluator_all.add_case(s_ranks, s_answer)
                    f_evaluator_all.add_case(f_ranks, f_answer)

                # Save per-sample result
                output = {
                    "scenario": scenario_name,
                    "sample_id": int(sample_dir.name),
                    "service": service,
                    "fault": fault,
                    "inject_time": inject_time,
                    "data_shape": list(df.shape),
                    "gamma": gamma,
                    "status": result.get("status", "unknown"),
                    "root_causes": ranks,
                    "elapsed_seconds": round(elapsed, 2),
                }
                (output_dir / "result.json").write_text(
                    json.dumps(output, indent=2, ensure_ascii=False)
                )

                s_hit = "S" if s_answer in s_ranks[:3] else "-"
                f_hit = "F" if f_answer in f_ranks[:3] else "-"
                print(f"{s_hit}{f_hit}  {len(ranks)}rc  {elapsed:.1f}s")
                success += 1

            except Exception as e:
                print(f"ERROR: {e}")

    avg_speed = round(total_elapsed / success, 2) if success else 0

    # =========== Print evaluation results (matching RCAEval main.py) ===========
    eval_data = {
        "service-fault": [],
        "top_1_service": [], "top_3_service": [], "top_5_service": [], "avg@5_service": [],
        "top_1_metric": [], "top_3_metric": [], "top_5_metric": [], "avg@5_metric": [],
    }

    print("--- Evaluation results ---")
    for name, s_ev, f_ev in [
        ("cpu", s_evaluator_cpu, f_evaluator_cpu),
        ("mem", s_evaluator_mem, f_evaluator_mem),
        ("io", s_evaluator_io, f_evaluator_io),
        ("socket", s_evaluator_socket, f_evaluator_socket),
        ("delay", s_evaluator_lat, f_evaluator_lat),
        ("loss", s_evaluator_loss, f_evaluator_loss),
    ]:
        eval_data["service-fault"].append(f"overall_{name}")
        eval_data["top_1_service"].append(s_ev.accuracy(1))
        eval_data["top_3_service"].append(s_ev.accuracy(3))
        eval_data["top_5_service"].append(s_ev.accuracy(5))
        eval_data["avg@5_service"].append(s_ev.average(5))
        eval_data["top_1_metric"].append(f_ev.accuracy(1))
        eval_data["top_3_metric"].append(f_ev.accuracy(3))
        eval_data["top_5_metric"].append(f_ev.accuracy(5))
        eval_data["avg@5_metric"].append(f_ev.average(5))

        display_name = "disk" if name == "io" else name
        if s_ev.average(5) is not None:
            print(f"Avg@5-{display_name.upper()}:".ljust(12), round(s_ev.average(5), 2))

    print("---")
    print("Avg speed:", avg_speed)

    # Save summary
    summary = {
        "algorithm": "IAF-RCL",
        "gamma": gamma,
        "total_samples": total,
        "success": success,
        "failed": total - success,
        "avg_speed": avg_speed,
        "eval_data": eval_data,
    }
    summary_path = dataset_dir / "rcd_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch-run RCD on RCAEval dataset")
    parser.add_argument("--dataset-dir", type=str,
                        default="data/RCAEval/online-boutique",
                        help="Path to the dataset root directory")
    parser.add_argument("--gamma", type=int, default=5, help="Gamma parameter (default: 5)")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    print(f"Dataset: {dataset_dir}")
    print(f"Gamma: {args.gamma}")
    run_eval(dataset_dir, args.gamma)


if __name__ == "__main__":
    main()