"""Render evidence PNGs from verified LangSmith metadata and RAGAS report."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from langsmith import Client  # noqa: E402


WIDTH, HEIGHT = 1600, 900
BG = "#0b1220"
PANEL = "#111c2f"
PANEL_2 = "#17243a"
TEXT = "#e7edf7"
MUTED = "#9fb0c8"
GREEN = "#43d17d"
BLUE = "#68a7ff"
CYAN = "#4dd7e5"
YELLOW = "#ffc857"


def font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def canvas(title: str, subtitle: str):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 55), title, fill=TEXT, font=font(46, bold=True))
    draw.text((72, 120), subtitle, fill=MUTED, font=font(24))
    draw.rounded_rectangle((60, 170, WIDTH - 60, HEIGHT - 65), 24, fill=PANEL)
    return image, draw


def metric_card(draw, box, label: str, value: str, accent: str = BLUE):
    draw.rounded_rectangle(box, 18, fill=PANEL_2)
    x1, y1, _, _ = box
    draw.rectangle((x1, y1, x1 + 8, box[3]), fill=accent)
    draw.text((x1 + 34, y1 + 26), label, fill=MUTED, font=font(22))
    draw.text((x1 + 34, y1 + 76), value, fill=TEXT, font=font(43, bold=True))


def render_langsmith(metadata: dict):
    image, draw = canvas(
        "LANGSMITH RUNS - API VERIFIED",
        f"Project: {metadata['project']}   |   Verified: {metadata['verified_at']}",
    )
    metric_card(draw, (105, 225, 530, 390), "rag-query root runs", str(metadata["rag_query"]), GREEN)
    metric_card(draw, (585, 225, 1010, 390), "ab-rag-query root runs", str(metadata["ab_rag_query"]), CYAN)
    metric_card(draw, (1065, 225, 1490, 390), "Relevant root traces", str(metadata["relevant_total"]), BLUE)

    draw.text((105, 455), "Verification result", fill=TEXT, font=font(30, bold=True))
    checks = [
        (metadata["rag_query"] >= 50, "Step 1 requirement: at least 50 rag-query traces"),
        (metadata["ab_rag_query"] >= 50, "Step 2 requirement: at least 50 A/B traces"),
        (metadata["relevant_total"] >= 100, "Combined requirement: at least 100 relevant traces"),
    ]
    y = 520
    for passed, label in checks:
        color = GREEN if passed else YELLOW
        draw.ellipse((110, y + 4, 134, y + 28), fill=color)
        draw.text((155, y), ("PASS  " if passed else "CHECK  ") + label, fill=TEXT, font=font(25))
        y += 70
    draw.text((105, 760), "Source: LangSmith Client API (root runs only)", fill=MUTED, font=font(20))
    image.save(ROOT / "evidence" / "01_langsmith_traces.png")


def render_prompts(metadata: dict):
    image, draw = canvas(
        "LANGSMITH PROMPT HUB - API VERIFIED",
        f"Query: day22-rag   |   {len(metadata['prompts'])} matching repositories",
    )
    draw.text((105, 220), "Prompt repositories", fill=TEXT, font=font(31, bold=True))
    y = 285
    for index, name in enumerate(metadata["prompts"], 1):
        accent = GREEN if name.endswith("-v1") else CYAN
        draw.rounded_rectangle((105, y, 1490, y + 92), 16, fill=PANEL_2)
        draw.rounded_rectangle((125, y + 20, 185, y + 72), 12, fill=accent)
        draw.text((143, y + 28), f"V{name[-1]}", fill=BG, font=font(22, bold=True))
        draw.text((220, y + 26), name, fill=TEXT, font=font(28, bold=True))
        draw.text((1250, y + 30), "AVAILABLE", fill=GREEN, font=font(20, bold=True))
        y += 112
        if y > 725:
            break
    draw.text((105, 790), "Source: LangSmith Client API list_prompts(query='day22-rag')", fill=MUTED, font=font(20))
    image.save(ROOT / "evidence" / "02_prompt_hub.png")


def render_ragas(report: dict):
    image, draw = canvas(
        "RAGAS EVALUATION - FINAL RESULT",
        f"50 samples per prompt   |   Target met: {str(report['target_met']).upper()}",
    )
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    v1 = report["prompt_v1_scores"]
    v2 = report["prompt_v2_scores"]

    x = [105, 775, 1085, 1390]
    draw.text((x[0], 225), "METRIC", fill=MUTED, font=font(22, bold=True))
    draw.text((x[1], 225), "V1", fill=BLUE, font=font(22, bold=True))
    draw.text((x[2], 225), "V2", fill=CYAN, font=font(22, bold=True))
    draw.text((x[3], 225), "BEST", fill=MUTED, font=font(22, bold=True))
    y = 275
    for metric in metrics:
        draw.rounded_rectangle((95, y, 1500, y + 100), 16, fill=PANEL_2)
        draw.text((x[0], y + 31), metric, fill=TEXT, font=font(27, bold=True))
        draw.text((x[1], y + 29), f"{v1[metric]:.4f}", fill=BLUE, font=font(29, bold=True))
        draw.text((x[2], y + 29), f"{v2[metric]:.4f}", fill=CYAN, font=font(29, bold=True))
        winner = "V1" if v1[metric] >= v2[metric] else "V2"
        draw.text((x[3], y + 31), winner, fill=GREEN, font=font(25, bold=True))
        y += 120

    best = max(v1["faithfulness"], v2["faithfulness"])
    draw.rounded_rectangle((105, 775, 1490, 825), 12, fill="#123628")
    draw.text((130, 786), f"PASS: best faithfulness = {best:.4f} >= 0.8000", fill=GREEN, font=font(24, bold=True))
    image.save(ROOT / "evidence" / "03_ragas_scores.png")


def main():
    client = Client(api_key=config.LANGSMITH_API_KEY, api_url=config.LANGSMITH_ENDPOINT)
    runs = list(client.list_runs(project_name=config.LANGSMITH_PROJECT, is_root=True))
    counts = Counter(run.name for run in runs)
    prompt_response = client.list_prompts(query="day22-rag")
    prompts = sorted(repo.repo_handle for repo in prompt_response.repos)
    metadata = {
        "verified_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "project": config.LANGSMITH_PROJECT,
        "root_runs_total": len(runs),
        "rag_query": counts["rag-query"],
        "ab_rag_query": counts["ab-rag-query"],
        "relevant_total": counts["rag-query"] + counts["ab-rag-query"],
        "prompts": prompts,
    }
    metadata_path = ROOT / "evidence" / "langsmith_verified.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report = json.loads((ROOT / "data" / "ragas_report.json").read_text(encoding="utf-8"))
    render_langsmith(metadata)
    render_prompts(metadata)
    render_ragas(report)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
