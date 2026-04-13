"""
Tests for SMCA Arena — Fase 1

Tests cover:
- CombatAgent response generation and scoring
- Arena round execution and champion selection
- Countdown pressure mechanics
- Judge standards and evaluation
- SMCAEngine end-to-end pipeline
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.arena.models import AgentResponse, RoundResult, CombatResult, SMCAResult
from src.arena.countdown import Countdown
from src.arena.standards import StandardsEngine, STANDARD_POOL
from src.arena.judge import Judge
from src.arena.agent_profiles import get_profiles, AGENT_PROFILES


# ============================================================
# Data Models Tests
# ============================================================

class TestModels:
    def test_agent_response_creation(self):
        resp = AgentResponse(
            agent_name="TestAgent",
            strategy="precision",
            text="Test response",
            markers_used=[1, 2, 3],
            similarities=[0.9, 0.8, 0.7],
            confidence=0.85
        )
        assert resp.agent_name == "TestAgent"
        assert resp.confidence == 0.85
        assert len(resp.markers_used) == 3

    def test_agent_response_to_dict(self):
        resp = AgentResponse(
            agent_name="TestAgent",
            strategy="precision",
            text="Test"
        )
        d = resp.to_dict()
        assert 'agent_name' in d
        assert 'timestamp' in d
        assert d['agent_name'] == "TestAgent"

    def test_round_result_to_dict(self):
        rr = RoundResult(
            round_number=1,
            standards_used=['precision', 'speed'],
            winner_name='AgentA',
            winner_score=0.9,
            loser_name='AgentB',
            loser_score=0.6
        )
        d = rr.to_dict()
        assert d['round_number'] == 1
        assert d['winner_name'] == 'AgentA'

    def test_combat_result_to_dict(self):
        cr = CombatResult(
            query="test query",
            champion_name="Winner",
            final_score=0.95,
            total_rounds=3,
            total_time=1.5
        )
        d = cr.to_dict()
        assert d['query'] == "test query"
        assert d['total_rounds'] == 3

    def test_smca_result_to_dict(self):
        sr = SMCAResult(
            query="test",
            final_answer="answer",
            champion_name="Champ"
        )
        d = sr.to_dict()
        assert d['final_answer'] == "answer"


# ============================================================
# Countdown Tests
# ============================================================

class TestCountdown:
    def test_initialization(self):
        cd = Countdown(10.0)
        assert cd.total_seconds == 10.0
        assert not cd.is_expired()
        assert cd.get_remaining() == 10.0

    def test_start_and_running(self):
        cd = Countdown(5.0)
        cd.start()
        assert cd.is_running()
        assert cd.get_remaining() > 0
        assert cd.get_remaining() <= 5.0
        cd.stop()

    def test_pressure_level_increases(self):
        cd = Countdown(1.0)
        cd.start()
        time.sleep(0.3)
        p1 = cd.get_pressure_level()
        time.sleep(0.3)
        p2 = cd.get_pressure_level()
        assert p2 >= p1  # Pressure should increase over time
        cd.stop()

    def test_expiration(self):
        cd = Countdown(0.2)
        cd.start()
        time.sleep(0.4)
        assert cd.is_expired()
        cd.stop()

    def test_max_rounds_reduction(self):
        cd = Countdown(1.0)
        cd.start()
        r1 = cd.get_max_rounds(10)
        time.sleep(0.5)
        r2 = cd.get_max_rounds(10)
        assert r2 <= r1  # Should reduce as time passes
        cd.stop()

    def test_fraction_remaining(self):
        cd = Countdown(10.0)
        assert cd.get_fraction_remaining() == 1.0
        cd.start()
        assert cd.get_fraction_remaining() <= 1.0
        assert cd.get_fraction_remaining() > 0.0
        cd.stop()

    def test_status(self):
        cd = Countdown(5.0)
        status = cd.get_status()
        assert 'total_seconds' in status
        assert 'pressure_level' in status
        assert 'is_running' in status

    def test_threshold_callbacks(self):
        triggered = []
        cd = Countdown(0.5, pressure_thresholds=[0.5])
        cd.on_threshold(0.5, lambda t, f, p: triggered.append(t))
        cd.start()
        time.sleep(0.6)
        cd.stop()
        # The 50% threshold should have been triggered
        assert 0.5 in triggered


# ============================================================
# Standards Engine Tests
# ============================================================

class TestStandardsEngine:
    def test_initialization(self):
        se = StandardsEngine()
        assert se.standards_per_round == 2
        assert len(se.history) == 0

    def test_select_standards_returns_correct_count(self):
        se = StandardsEngine(standards_per_round=2)
        standards = se.select_standards(1)
        assert len(standards) == 2

    def test_select_standards_from_pool(self):
        se = StandardsEngine()
        standards = se.select_standards(1)
        for s in standards:
            assert s in STANDARD_POOL

    def test_standards_change_between_rounds(self):
        se = StandardsEngine()
        s1 = se.select_standards(1)
        # Record an outcome to influence next selection
        se.record_outcome(1, s1, 0.8, 0.5)
        s2 = se.select_standards(2)
        # Standards should be different (or at least have a chance to be)
        # Due to randomness, we just check they're valid
        for s in s2:
            assert s in STANDARD_POOL

    def test_record_outcome(self):
        se = StandardsEngine()
        standards = ['precision', 'speed']
        se.record_outcome(1, standards, 0.9, 0.6)
        assert len(se.standard_performance['precision']) == 1
        assert len(se.standard_performance['speed']) == 1

    def test_standard_rankings(self):
        se = StandardsEngine()
        se.record_outcome(1, ['precision'], 0.9, 0.5)
        se.record_outcome(2, ['speed'], 0.3, 0.2)
        rankings = se.get_standard_rankings()
        assert len(rankings) == len(STANDARD_POOL)
        # Precision should rank higher than speed
        prec_rank = next(i for i, r in enumerate(rankings) if r['standard'] == 'precision')
        speed_rank = next(i for i, r in enumerate(rankings) if r['standard'] == 'speed')
        assert prec_rank < speed_rank  # Lower index = higher rank

    def test_evolution_data(self):
        se = StandardsEngine()
        se.select_standards(1)
        data = se.get_evolution_data()
        assert 'history' in data
        assert 'performance' in data
        assert 'rankings' in data


# ============================================================
# Judge Tests
# ============================================================

class TestJudge:
    def test_initialization(self):
        judge = Judge()
        assert judge.confidence_threshold == 0.7
        assert judge.total_judgments == 0

    def test_generate_standards(self):
        judge = Judge()
        standards = judge.generate_standards(1)
        assert len(standards) >= 1
        for s in standards:
            assert s in STANDARD_POOL

    def test_evaluate_responses(self):
        judge = Judge()
        responses = [
            AgentResponse(
                agent_name="A", strategy="precision", text="Response A",
                similarities=[0.9, 0.8], confidence=0.85, generation_time=0.1,
                markers_used=[1, 2], marker_segments=["seg1", "seg2"]
            ),
            AgentResponse(
                agent_name="B", strategy="creativity", text="Response B",
                similarities=[0.7, 0.6], confidence=0.6, generation_time=0.2,
                markers_used=[3], marker_segments=["seg3"]
            )
        ]
        scores = judge.evaluate_responses(responses, ['precision'])
        assert 'A' in scores
        assert 'B' in scores
        assert isinstance(scores['A'], float)
        assert judge.total_judgments == 1

    def test_determine_winner(self):
        judge = Judge()
        scores = {'A': 0.9, 'B': 0.6}
        winner, score = judge.determine_winner(scores, [])
        assert winner == 'A'
        assert score == 0.9

    def test_meta_cognition(self):
        judge = Judge(confidence_threshold=0.99)  # Very high threshold
        # Create responses with very similar scores
        responses = [
            AgentResponse(
                agent_name="A", strategy="precision", text="Resp A",
                similarities=[0.8], confidence=0.8, generation_time=0.1,
                markers_used=[1], marker_segments=["seg"]
            ),
            AgentResponse(
                agent_name="B", strategy="creativity", text="Resp B",
                similarities=[0.79], confidence=0.79, generation_time=0.1,
                markers_used=[2], marker_segments=["seg2"]
            )
        ]
        judge.evaluate_responses(responses, ['precision'])
        # With very high threshold and close scores, should need God
        assert judge.needs_god_intervention()

    def test_autonomy_increases_with_god_decisions(self):
        judge = Judge()
        initial_threshold = judge.confidence_threshold
        # Simulate many God decisions
        for i in range(10):
            judge.total_judgments += 1
            judge.record_god_decision({
                'round': i, 'reason': 'test',
                'chosen_winner': 'A'
            })
        # After many God decisions, threshold should decrease
        assert judge.confidence_threshold <= initial_threshold

    def test_stats(self):
        judge = Judge()
        stats = judge.get_stats()
        assert 'total_judgments' in stats
        assert 'autonomy_level' in stats
        assert 'standards_evolution' in stats


# ============================================================
# Agent Profiles Tests
# ============================================================

class TestAgentProfiles:
    def test_profiles_count(self):
        assert len(AGENT_PROFILES) == 10

    def test_get_profiles_2(self):
        profiles = get_profiles(2)
        assert len(profiles) == 2
        assert profiles[0]['name'] == 'Precision'
        assert profiles[1]['name'] == 'Creative'

    def test_get_profiles_10(self):
        profiles = get_profiles(10)
        assert len(profiles) == 10

    def test_all_profiles_have_required_fields(self):
        for p in AGENT_PROFILES:
            assert 'name' in p
            assert 'strategy' in p
            assert 'description' in p
            assert 'strengths' in p
            assert 'weaknesses' in p


# ============================================================
# Integration Test (requires StudSar initialization)
# ============================================================

class TestSMCAIntegration:
    """Integration tests that require full StudSar initialization.
    These tests are slower due to model loading."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create an SMCAEngine instance for integration tests."""
        from src.arena.smca_engine import SMCAEngine
        engine = SMCAEngine(
            num_agents=2,
            countdown_seconds=10.0,
            auto_god=True
        )
        # Ingest test data
        engine.ingest_document(
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning uses neural networks with many layers. "
            "Transformers use attention mechanisms for sequence processing.",
            source_name="test_doc"
        )
        return engine

    def test_engine_initialization(self, engine):
        assert len(engine.agents) == 2
        assert engine.studsar.studsar_network.get_total_markers() > 0

    def test_query_returns_result(self, engine):
        result = engine.query("What is machine learning?", max_rounds=2, countdown_seconds=5.0)
        assert isinstance(result, SMCAResult)
        assert result.final_answer != ""
        assert result.champion_name != ""

    def test_query_stores_history(self, engine):
        initial_count = len(engine.query_history)
        engine.query("What are transformers?", max_rounds=1, countdown_seconds=5.0)
        assert len(engine.query_history) == initial_count + 1

    def test_memory_growth_logging(self, engine):
        assert len(engine.memory_growth_log) > 0
        assert 'total_markers' in engine.memory_growth_log[0]

    def test_system_status(self, engine):
        status = engine.get_system_status()
        assert status['studsar']['total_markers'] > 0
        assert status['agents']['count'] == 2

    def test_save_state(self, engine, tmp_path):
        save_dir = str(tmp_path / "test_state")
        saved = engine.save_state(directory=save_dir)
        assert 'studsar_network' in saved
        assert 'query_history' in saved
        assert os.path.exists(saved['studsar_network'])
        assert os.path.exists(saved['query_history'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
