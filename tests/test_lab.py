"""Kiểm thử ngoại tuyến cho các phần không cần API key."""

import importlib
import json
import sys
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


class FakeVectorstore:
    def as_retriever(self, **_kwargs):
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(
            lambda _question: [
                Document(page_content="RAG uses retrieval and generation."),
                Document(page_content="Retrieved context grounds the answer."),
            ]
        )


class DeterministicEmbeddings(Embeddings):
    """Embedding nhỏ, tất định để kiểm tra FAISS mà không gọi API."""

    @staticmethod
    def _embed(text):
        vector = [0.0] * 16
        for index, byte in enumerate(text.encode("utf-8")):
            vector[index % len(vector)] += float(byte)
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step1 = importlib.import_module("01_langsmith_rag_pipeline")
        cls.step2 = importlib.import_module("02_prompt_hub_ab_routing")
        cls.step3 = importlib.import_module("03_ragas_evaluation")
        cls.step4 = importlib.import_module("04_guardrails_validator")
        cls.qa = importlib.import_module("qa_pairs")
        cls.data_loader = importlib.import_module("utils.data_loader")

    def test_dataset_has_exactly_fifty_questions(self):
        self.assertEqual(len(self.qa.SAMPLE_QUESTIONS), 50)
        self.assertEqual(len(self.qa.QA_PAIRS), 50)

    def test_knowledge_base_chunks_and_faiss_index(self):
        text = self.data_loader.load_knowledge_base()
        chunks = self.data_loader.split_text(
            text, chunk_size=500, chunk_overlap=50
        )
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))

        vectorstore = self.data_loader.build_vectorstore(
            chunks, DeterministicEmbeddings()
        )
        docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke("RAG")
        self.assertEqual(len(docs), 3)

    def test_step1_lcel_chain_returns_text(self):
        from langchain_core.runnables import RunnableLambda

        original_get_llm = self.step1.get_llm
        self.step1.get_llm = lambda: RunnableLambda(lambda _prompt: "grounded answer")
        try:
            chain, _retriever = self.step1.build_rag_chain(FakeVectorstore())
            self.assertEqual(chain.invoke("What is RAG?"), "grounded answer")
        finally:
            self.step1.get_llm = original_get_llm

    def test_prompt_versions_are_distinct_and_use_context(self):
        self.assertNotEqual(self.step2.SYSTEM_V1, self.step2.SYSTEM_V2)
        self.assertEqual(self.step2.SYSTEM_V1, self.step3.SYSTEM_V1)
        self.assertEqual(self.step2.SYSTEM_V2, self.step3.SYSTEM_V2)
        for prompt in (self.step2.PROMPT_V1, self.step2.PROMPT_V2):
            self.assertEqual(set(prompt.input_variables), {"context", "question"})

    def test_ab_routing_is_deterministic_and_uses_both_versions(self):
        first = [
            self.step2.get_prompt_version(f"req-{index:04d}")
            for index in range(50)
        ]
        second = [
            self.step2.get_prompt_version(f"req-{index:04d}")
            for index in range(50)
        ]
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {self.step2.PROMPT_V1_NAME, self.step2.PROMPT_V2_NAME},
        )

    def test_run_rag_preserves_context_list(self):
        from langchain_core.runnables import RunnableLambda

        retriever = FakeVectorstore().as_retriever()
        llm = RunnableLambda(lambda _prompt: "RAG combines retrieval and generation.")
        result = self.step3.run_rag(
            retriever, llm, self.step3.PROMPT_V1, "What is RAG?"
        )
        self.assertIsInstance(result["contexts"], list)
        self.assertEqual(len(result["contexts"]), 2)
        self.assertIn("retrieval", result["answer"])

    def test_ragas_dataset_schema(self):
        rows = [{
            "question": "What is RAG?",
            "answer": "Retrieval plus generation.",
            "contexts": ["RAG uses retrieval and generation."],
            "reference": "RAG combines retrieval and generation.",
        }]
        dataset = self.step3.build_ragas_dataset(rows)
        sample = dataset.samples[0]
        self.assertEqual(sample.user_input, rows[0]["question"])
        self.assertEqual(sample.response, rows[0]["answer"])
        self.assertEqual(sample.retrieved_contexts, rows[0]["contexts"])
        self.assertEqual(sample.reference, rows[0]["reference"])

    def test_pii_guard_redacts_all_supported_types(self):
        guard = self.step4.Guard().use(
            self.step4.PIIDetector(on_fail=self.step4.OnFailAction.FIX)
        )
        value = (
            "Email a@example.com, call (555) 867-5309, SSN 123-45-6789, "
            "card 4532 1234 5678 9010."
        )
        output = guard.validate(value).validated_output
        for marker in ("EMAIL", "PHONE", "SSN", "CREDIT_CARD"):
            self.assertIn(f"[{marker}_REDACTED]", output)

    def test_json_guard_repairs_and_has_safe_fallback(self):
        guard = self.step4.Guard().use(
            self.step4.JSONFormatter(on_fail=self.step4.OnFailAction.FIX)
        )
        repaired = guard.validate("```json\n{'ok': true,}\n```").validated_output
        self.assertEqual(json.loads(repaired), {"ok": True})

        fallback = guard.validate("not json {]").validated_output
        self.assertEqual(json.loads(fallback)["error"], "invalid_json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
