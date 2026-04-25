import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.arena.post_retrieval_executor import PostRetrievalExecutor


class TestPostRetrievalExecutorEntity(unittest.TestCase):
    def test_extracts_project_with_letter_suffix(self):
        executor = PostRetrievalExecutor()
        segments = [
            {
                "index": 0,
                "text": "[SEGMENTO 0] Progetto A: Fase iniziale=concept, Fase finale=lancio, Durata=12 mesi.",
            },
            {
                "index": 1,
                "text": "[SEGMENTO 1] Progetto B: Fase iniziale=sviluppo, Fase finale=testing, Durata=24 mesi.",
            },
            {
                "index": 2,
                "text": "[SEGMENTO 2] Progetto C: Fase iniziale=pianificazione, Fase finale=deployment, Durata=35 mesi.",
            },
        ]
        facts = executor.extract_numerical_facts(segments)
        by_value = {int(f.value): f.entity for f in facts if f.unit == "mesi"}

        self.assertEqual(by_value.get(12), "Progetto A")
        self.assertEqual(by_value.get(24), "Progetto B")
        self.assertEqual(by_value.get(35), "Progetto C")

    def test_execute_max_returns_project_name(self):
        executor = PostRetrievalExecutor()
        segments = [
            {
                "index": 0,
                "text": "[SEGMENTO 0] Progetto A: Fase iniziale=concept, Fase finale=lancio, Durata=12 mesi.",
            },
            {
                "index": 1,
                "text": "[SEGMENTO 1] Progetto B: Fase iniziale=sviluppo, Fase finale=testing, Durata=24 mesi.",
            },
            {
                "index": 2,
                "text": "[SEGMENTO 2] Progetto C: Fase iniziale=pianificazione, Fase finale=deployment, Durata=35 mesi.",
            },
        ]
        res = executor.execute("Qual è il progetto più lungo?", segments)
        self.assertEqual(res["operation"], "max")
        self.assertEqual(res["result"], "Progetto C")

    def test_execute_max_tie_breaks_by_earliest_segment(self):
        executor = PostRetrievalExecutor()
        segments = [
            {
                "index": 0,
                "text": "[SEGMENTO 0] Progetto A: Durata=35 mesi.",
            },
            {
                "index": 1,
                "text": "[SEGMENTO 1] Progetto B: Durata=35 mesi.",
            },
            {
                "index": 2,
                "text": "[SEGMENTO 2] Progetto C: Durata=10 mesi.",
            },
        ]
        res = executor.execute("Qual è il progetto più lungo?", segments)
        self.assertEqual(res["result"], "Progetto A")

    def test_extracts_entity_after_number_when_needed(self):
        executor = PostRetrievalExecutor()
        segments = [
            {
                "index": 0,
                "text": "[SEGMENTO 0] Durata=12 mesi per Progetto A.",
            }
        ]
        facts = executor.extract_numerical_facts(segments)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].entity, "Progetto A")


if __name__ == "__main__":
    unittest.main()
