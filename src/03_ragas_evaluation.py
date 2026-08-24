"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import csv
import json
import warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "You are a concise AI assistant. Answer using only the provided context. "
    "Give a direct answer in 2-4 sentences. If the context is insufficient, "
    "say that you do not know; do not add unsupported facts.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an AI and machine-learning expert. Use only the provided context. "
    "Write a clear, structured answer in 3-5 sentences: start with the direct "
    "answer, then explain the most relevant supporting facts. Explicitly state "
    "when the context is insufficient and never invent information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]


def rag_outputs_cache_path() -> Path:
    """Cache đáp án/context để có thể chạy lại evaluator mà không gọi RAG lại."""
    return Path(__file__).parent.parent / ".cache" / "ragas_outputs.json"


def load_rag_outputs_cache() -> dict:
    path = rag_outputs_cache_path()
    if not path.exists():
        # Di chuyển mềm cache cũ có model trong tên sang định dạng dùng chung.
        legacy = sorted(path.parent.glob("ragas_outputs_*.json"))
        if not legacy:
            return {}
        path = legacy[-1]
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        version: rows
        for version, rows in cached.items()
        if version in PROMPTS
        and isinstance(rows, list)
        and len(rows) <= len(QA_PAIRS)
    }


def save_rag_outputs_cache(outputs: dict) -> Path:
    path = rag_outputs_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(
    vectorstore,
    prompt_version: str,
    existing_results: list | None = None,
    checkpoint=None,
) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    if prompt_version not in PROMPTS:
        raise ValueError(f"Prompt version không hợp lệ: {prompt_version}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    existing_by_question = {
        row["question"]: row for row in (existing_results or [])
    }
    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        if qa["question"] in existing_by_question:
            results.append(existing_by_question[qa["question"]])
            print(f"  [{i:02d}/50] [cache] {qa['question'][:52]}")
            continue

        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        if checkpoint:
            checkpoint(results)
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


def seed_v2_cache_from_ab_log(vectorstore, cached_outputs: dict) -> int:
    """Tận dụng các answer V2 thật từ Bước 2 khi evaluator cần resume."""
    if cached_outputs.get("v2"):
        return 0

    log_path = Path(__file__).parent.parent / "evidence" / "02_ab_routing_log.txt"
    if not log_path.exists():
        return 0

    references = {qa["question"]: qa["reference"] for qa in QA_PAIRS}
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    seeded = []
    with log_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            question = row.get("question", "")
            if row.get("version") != "v2" or question not in references:
                continue
            docs = retriever.invoke(question)
            seeded.append({
                "question": question,
                "reference": references[question],
                "answer": row["answer"],
                "contexts": [doc.page_content for doc in docs],
            })

    if seeded:
        cached_outputs["v2"] = seeded
        save_rag_outputs_cache(cached_outputs)
        print(f"♻️  Đã phục hồi {len(seeded)} đáp án V2 từ log A/B")
    return len(seeded)


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(
            timeout=900,
            max_retries=10,
            max_wait=60,
            max_workers=4,
        ),
        batch_size=4,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in METRIC_NAMES:
        raw = result[key]
        values = [
            float(value)
            for value in raw
            if value is not None and not np.isnan(float(value))
        ]
        if not values:
            raise RuntimeError(f"RAGAS không trả về điểm hợp lệ cho metric '{key}'")
        scores[key] = float(np.mean(values))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


def write_evidence_analysis(v1_scores: dict, v2_scores: dict) -> Path:
    """Cập nhật phân tích V1/V2 trong evidence README bằng điểm chạy thật."""
    readme_path = Path(__file__).parent.parent / "evidence" / "README.md"
    start_marker = "<!-- RAGAS_ANALYSIS_START -->"
    end_marker = "<!-- RAGAS_ANALYSIS_END -->"

    faith_winner = (
        "V1" if v1_scores["faithfulness"] >= v2_scores["faithfulness"] else "V2"
    )
    v1_wins = sum(v1_scores[name] >= v2_scores[name] for name in METRIC_NAMES)
    v2_wins = len(METRIC_NAMES) - v1_wins
    analysis = (
        f"{start_marker}\n"
        f"Kết quả thực nghiệm cho thấy **{faith_winner}** có faithfulness cao hơn "
        f"(V1={v1_scores['faithfulness']:.4f}, "
        f"V2={v2_scores['faithfulness']:.4f}). V1 thắng {v1_wins}/4 metric và "
        f"V2 thắng {v2_wins}/4 metric. V1 ưu tiên câu trả lời ngắn gọn, còn V2 "
        "yêu cầu giải thích có cấu trúc; chênh lệch điểm có thể phản ánh lượng nội "
        "dung bổ sung mà mỗi prompt tạo ra từ cùng retrieved context.\n"
        f"{end_marker}"
    )

    current = readme_path.read_text(encoding="utf-8")
    start = current.index(start_marker)
    end = current.index(end_marker, start) + len(end_marker)
    updated = current[:start] + analysis + current[end:]
    readme_path.write_text(updated, encoding="utf-8")
    return readme_path


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2. Cache giúp chạy lại evaluator khi
    # mạng/quota gián đoạn mà không phải sinh lại 100 đáp án.
    cached_outputs = load_rag_outputs_cache()
    seed_v2_cache_from_ab_log(vectorstore, cached_outputs)
    for version in PROMPTS:
        if len(cached_outputs.get(version, [])) == len(QA_PAIRS):
            print(f"♻️  Đang dùng 50 đáp án {version} đã cache")
            continue

        def checkpoint(rows, selected_version=version):
            cached_outputs[selected_version] = list(rows)
            save_rag_outputs_cache(cached_outputs)

        cached_outputs[version] = collect_rag_outputs(
            vectorstore,
            version,
            existing_results=cached_outputs.get(version),
            checkpoint=checkpoint,
        )
        cache_path = save_rag_outputs_cache(cached_outputs)
        print(f"💾 Đã cache đáp án {version} tại {cache_path}")

    v1_results = cached_outputs["v1"]
    v2_results = cached_outputs["v2"]

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples_per_prompt": len(QA_PAIRS),
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_json = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    report_path.write_text(report_json, encoding="utf-8")

    evidence_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(report_json, encoding="utf-8")
    analysis_path = write_evidence_analysis(v1_scores, v2_scores)
    print(f"💾 Đã lưu báo cáo vào {report_path}")
    print(f"💾 Đã sao chép báo cáo vào {evidence_path}")
    print(f"📝 Đã cập nhật phân tích V1/V2 trong {analysis_path}")


if __name__ == "__main__":
    main()
