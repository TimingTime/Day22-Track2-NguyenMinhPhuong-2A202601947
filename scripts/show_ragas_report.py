"""In bảng RAGAS đã lưu, dùng được trong PowerShell, CMD và Bash."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data" / "ragas_report.json"


def main():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    v1 = report["prompt_v1_scores"]
    v2 = report["prompt_v2_scores"]
    metrics = (
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    )

    print("=" * 65)
    print("              RAGAS EVALUATION - FINAL REPORT")
    print("=" * 65)
    print(f"Samples per prompt: {report['samples_per_prompt']}\n")
    print(f"{'Metric':22} {'V1':>9} {'V2':>9} {'Winner':>9}")
    print("-" * 65)
    for metric in metrics:
        winner = "Tie" if v1[metric] == v2[metric] else ("V1" if v1[metric] > v2[metric] else "V2")
        print(f"{metric:22} {v1[metric]:>9.4f} {v2[metric]:>9.4f} {winner:>9}")
    print("-" * 65)
    status = "PASS" if report["target_met"] else "FAIL"
    print(f"Target faithfulness >= 0.8: {status}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
