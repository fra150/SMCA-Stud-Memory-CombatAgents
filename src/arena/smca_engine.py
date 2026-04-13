"""
SMCAEngine — The Main Orchestrator for Stud Memory Combat Agents.

This is the entry point for the entire SMCA system. It coordinates:
- StudSar shared memory (the living core)
- Combat Agents with specialized strategies
- The Arena where agents compete
- The Judge with emergent standards
- The Countdown evolutionary pressure
- The God Protocol for human escalation

StudSar is the pivot — every agent reads from it, every outcome is stored in it.
"""

import time
import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..managers.manager import StudSarManager
from .agent import CombatAgent
from .arena import Arena
from .judge import Judge
from .countdown import Countdown
from .god_protocol import GodProtocol
from .agent_profiles import get_profiles
from .models import SMCAResult, CombatResult


class SMCAEngine:
    """Main orchestrator for the SMCA system.
    
    The engine manages the full pipeline:
    1. Document ingestion into StudSar
    2. Agent creation with specialized strategies
    3. Arena combat with countdown pressure
    4. Judge arbitration with emergent standards
    5. God escalation when needed
    6. Final answer delivery
    """
    
    def __init__(self, studsar_manager: Optional[StudSarManager] = None,
                 num_agents: int = 2,
                 countdown_seconds: float = 60.0,
                 judge_confidence_threshold: float = 0.7,
                 auto_god: bool = True,
                 embedding_generator=None,
                 device=None,
                 enable_judge_memory: bool = True,
                 allow_empty_memory: bool = False,
                 enable_red_agent: bool = False,
                 red_alpha: float = 0.35,
                 red_tau: float = 0.08,
                 red_write_negative_markers: bool = False):
        """Initialize the SMCA Engine.
        
        Args:
            studsar_manager: Shared StudSarManager. Created if None.
            num_agents: Number of agents (2 for Fase 1, up to 10 for Fase 3)
            countdown_seconds: Default countdown duration per combat
            judge_confidence_threshold: Judge escalation threshold
            auto_god: If True, God protocol auto-resolves (no human needed)
        """
        print("\n" + "=" * 60)
        print("  SMCA ENGINE — Stud Memory Combat Agents")
        print("  Initializing cognitive arena system...")
        print("=" * 60)
        
        # StudSar is the PIVOT — the living core
        self.studsar = studsar_manager if studsar_manager else StudSarManager(
            embedding_generator=embedding_generator,
            device=device
        )
        self.allow_empty_memory = allow_empty_memory
        
        self.judge_memory = None
        if enable_judge_memory:
            self.judge_memory = StudSarManager(
                embedding_generator=self.studsar.embedding_generator,
                device=self.studsar.device,
                initial_capacity=256
            )

        # Judge with its own memory perspective
        self.judge = Judge(
            studsar_manager=self.judge_memory,
            confidence_threshold=judge_confidence_threshold
        )
        
        # Create agents from profiles
        profiles = get_profiles(num_agents)
        self.agents: List[CombatAgent] = []
        for profile in profiles:
            agent = CombatAgent(
                name=profile['name'],
                strategy=profile['strategy'],
                studsar_manager=self.studsar
            )
            self.agents.append(agent)
            print(f"  ✦ Agent created: {profile['name']} ({profile['strategy']})")

        self.enable_red_agent = enable_red_agent
        self.red_alpha = red_alpha
        self.red_tau = red_tau
        self.red_write_negative_markers = red_write_negative_markers
        self.red_agent: Optional[CombatAgent] = None
        if enable_red_agent:
            self.red_agent = CombatAgent(
                name="Ziora",
                strategy="red",
                studsar_manager=self.studsar
            )
            print(f"  ✦ Red Agent enabled: Ziora (adversarial)")
        
        # God Protocol
        self.god = GodProtocol(self.judge, auto_resolve=auto_god)

        # Arena
        self.countdown_seconds = countdown_seconds
        self.arena = Arena(self.agents, self.judge, god_protocol=self.god)
        
        # History for visualizations
        self.query_history: List[SMCAResult] = []
        self.memory_growth_log: List[Dict[str, Any]] = []
        self.arena_performance_log: List[Dict[str, Any]] = []
        
        # Record initial memory state
        self._log_memory_state("initialization")
        
        print(f"\n  System ready:")
        print(f"    StudSar markers: {self.studsar.studsar_network.get_total_markers()}")
        if self.judge_memory:
            print(f"    Judge markers: {self.judge_memory.studsar_network.get_total_markers()}")
        print(f"    Agents: {len(self.agents)}")
        print(f"    Countdown: {countdown_seconds}s")
        print(f"    Judge threshold: {judge_confidence_threshold}")
        print(f"    God mode: {'auto' if auto_god else 'human'}")
        print("=" * 60 + "\n")
    
    def ingest_document(self, text: str, emotion: Optional[str] = None,
                       source_name: str = "document") -> Dict[str, Any]:
        """Ingest a document into the shared StudSar memory.
        
        This is how the system learns — all agents share this memory.
        
        Args:
            text: Document text to ingest
            emotion: Optional emotional tag for the content
            source_name: Name/identifier for the source document
            
        Returns:
            Dict with ingestion statistics
        """
        print(f"\n📥 Ingesting document: '{source_name}'")
        markers_before = self.studsar.studsar_network.get_total_markers()
        
        self.studsar.build_network_from_text(
            text,
            default_emotion=emotion or 'informative'
        )
        
        markers_after = self.studsar.studsar_network.get_total_markers()
        new_markers = markers_after - markers_before
        
        # Log memory growth
        self._log_memory_state(f"ingestion:{source_name}")
        
        stats = {
            'source': source_name,
            'markers_before': markers_before,
            'markers_after': markers_after,
            'new_markers': new_markers,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  ✓ Ingested {new_markers} new markers (total: {markers_after})")
        return stats
    
    def ingest_text(self, text: str, emotion: Optional[str] = None) -> int:
        """Quick ingest a single text segment into StudSar.
        
        Args:
            text: Text to add to memory
            emotion: Optional emotional tag
            
        Returns:
            Marker ID of the new segment
        """
        marker_id = self.studsar.update_network(text, emotion=emotion)
        self._log_memory_state("text_addition")
        return marker_id
    
    def query(self, question: str, max_rounds: int = 3,
             countdown_seconds: Optional[float] = None,
             standards_override: Optional[List[List[str]]] = None,
             selection_mode: str = "champion") -> SMCAResult:
        """Process a query through the full SMCA pipeline.
        
        This is the main method — it orchestrates the entire system:
        1. StudSar provides the memory foundation
        2. Agents compete in the Arena
        3. Judge arbitrates with emergent standards
        4. Countdown forces convergence
        5. Best answer emerges through competition
        
        Args:
            question: The query to process
            max_rounds: Maximum combat rounds
            countdown_seconds: Override countdown duration
            
        Returns:
            SMCAResult with the final answer and metadata
        """
        start_time = time.time()
        cd_seconds = countdown_seconds or self.countdown_seconds
        
        print(f"\n{'*' * 60}")
        print(f"  SMCA QUERY: {question}")
        print(f"{'*' * 60}")
        
        # Check if we have memory to work with
        total_markers = self.studsar.studsar_network.get_total_markers()
        if total_markers == 0 and not self.allow_empty_memory:
            return SMCAResult(
                query=question,
                final_answer="[SMCA] No memory available. Please ingest documents first.",
                champion_name="None",
                memory_stats={'total_markers': 0},
                total_processing_time=time.time() - start_time
            )
        
        # Reset arena for fresh combat
        self.arena.reset()
        
        # Run the combat
        combat_result = self.arena.run_combat(
            query=question,
            max_rounds=max_rounds,
            countdown_seconds=cd_seconds,
            standards_override=standards_override,
            selection_mode=selection_mode
        )
        
        # Extract final answer
        if combat_result.champion_response:
            final_answer = combat_result.champion_response.text
        else:
            final_answer = f"[SMCA] Combat completed but no response generated."

        base_confidence = self.judge.get_confidence()
        resilience_score = 0.0
        final_confidence = base_confidence

        if self.enable_red_agent and self.red_agent and combat_result.champion_response:
            k = min(10, max(total_markers, 0))
            marker_ids, sims, segs = self.studsar.search_with_reputation(question, k=k, reputation_weight=1.0)
            evidence_segments = list(combat_result.champion_response.marker_segments or [])
            for s in segs or []:
                if s and s not in evidence_segments:
                    evidence_segments.append(s)

            red_response = self.red_agent.generate_response(
                query=question,
                standards=['resilience'],
                context={
                    'champion_text': combat_result.champion_response.text,
                    'evidence_segments': evidence_segments,
                    'tau': self.red_tau,
                    'segments_override': evidence_segments,
                    'similarities_override': list(sims or []),
                    'marker_ids_override': list(marker_ids or []),
                }
            )

            red_metrics = (red_response.metadata or {}).get('red_metrics') or {}
            resilience_score = self.judge.compute_champion_resilience_score(red_metrics)
            final_confidence = self.judge.modulate_final_confidence(base_confidence, resilience_score, self.red_alpha)

            if self.red_write_negative_markers:
                findings = list((red_metrics or {}).get('findings') or [])
                for finding in findings[:3]:
                    self.studsar.add_negative_shadow(
                        query=question,
                        finding=str(finding or ""),
                        champion=combat_result.champion_name,
                        timestamp=datetime.now().isoformat()
                    )
        
        # Build SMCA result
        total_time = time.time() - start_time
        
        result = SMCAResult(
            query=question,
            final_answer=final_answer,
            champion_name=combat_result.champion_name,
            combat_result=combat_result,
            memory_stats={
                'total_markers': total_markers,
                'markers_accessed': sum(
                    len(r.markers_used) 
                    for round_r in combat_result.rounds 
                    for r in round_r.all_responses
                )
            },
            judge_confidence=final_confidence,
            base_judge_confidence=base_confidence,
            champion_resilience_score=resilience_score,
            standards_evolution=[r.standards_used for r in combat_result.rounds],
            god_was_consulted=combat_result.god_interventions > 0,
            total_processing_time=total_time
        )
        
        # Store in history
        self.query_history.append(result)
        self._log_arena_performance(result)
        self._log_memory_state(f"query:{question[:30]}")
        try:
            self.studsar.promote_shadow_negatives(min_confirmations=2, blocked_keys=set())
        except Exception:
            pass
        
        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  SMCA ANSWER")
        print(f"  Champion: {result.champion_name}")
        print(f"  Confidence: {result.judge_confidence:.3f}")
        if self.enable_red_agent:
            print(f"  Resilience (CRS): {result.champion_resilience_score:.3f} (alpha={self.red_alpha:.2f})")
        print(f"  Rounds: {combat_result.total_rounds}")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Standards evolution: {result.standards_evolution}")
        print(f"{'=' * 60}")
        print(f"\n{final_answer}\n")
        
        return result
    
    def save_state(self, directory: str = "smca_state") -> Dict[str, str]:
        """Save the complete SMCA state to disk.
        
        Saves:
        - StudSar network state (.pth)
        - Arena history (JSON)
        - Agent statistics (JSON)
        - Memory growth log (JSON)
        - Arena performance log (JSON)
        
        Args:
            directory: Directory to save state files
            
        Returns:
            Dict mapping component -> filepath
        """
        os.makedirs(directory, exist_ok=True)
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save StudSar network
        pth_path = os.path.join(directory, f"studsar_network_{timestamp}.pth")
        self.studsar.save(pth_path)
        saved_files['studsar_network'] = pth_path

        if self.judge_memory:
            judge_pth_path = os.path.join(directory, f"judge_studsar_network_{timestamp}.pth")
            self.judge_memory.save(judge_pth_path)
            saved_files['judge_studsar_network'] = judge_pth_path
        
        # 2. Save query history
        history_path = os.path.join(directory, f"query_history_{timestamp}.json")
        history_data = [r.to_dict() for r in self.query_history]
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
        saved_files['query_history'] = history_path
        
        # 3. Save agent statistics
        agents_path = os.path.join(directory, f"agent_stats_{timestamp}.json")
        agent_data = {a.name: a.get_stats() for a in self.agents}
        with open(agents_path, 'w', encoding='utf-8') as f:
            json.dump(agent_data, f, indent=2, ensure_ascii=False, default=str)
        saved_files['agent_stats'] = agents_path
        
        # 4. Save memory growth log
        growth_path = os.path.join(directory, f"memory_growth_{timestamp}.json")
        with open(growth_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory_growth_log, f, indent=2, ensure_ascii=False, default=str)
        saved_files['memory_growth'] = growth_path
        
        # 5. Save arena performance log
        perf_path = os.path.join(directory, f"arena_performance_{timestamp}.json")
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(self.arena_performance_log, f, indent=2, ensure_ascii=False, default=str)
        saved_files['arena_performance'] = perf_path
        
        # 6. Save Judge stats
        judge_path = os.path.join(directory, f"judge_stats_{timestamp}.json")
        with open(judge_path, 'w', encoding='utf-8') as f:
            json.dump(self.judge.get_stats(), f, indent=2, ensure_ascii=False, default=str)
        saved_files['judge_stats'] = judge_path
        
        # 7. Save God Protocol stats 
        god_path = os.path.join(directory, f"god_protocol_{timestamp}.json")
        with open(god_path, 'w', encoding='utf-8') as f:
            json.dump(self.god.get_stats(), f, indent=2, ensure_ascii=False, default=str)
        saved_files['god_protocol'] = god_path
        
        print(f"\n💾 SMCA state saved to '{directory}/':")
        for component, path in saved_files.items():
            print(f"   {component}: {os.path.basename(path)}")
        
        return saved_files
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'studsar': {
                'total_markers': self.studsar.studsar_network.get_total_markers(),
                'embedding_dim': self.studsar.studsar_network.embedding_dim
            },
            'judge_memory': {
                'total_markers': self.judge_memory.studsar_network.get_total_markers() if self.judge_memory else 0,
                'embedding_dim': self.judge_memory.studsar_network.embedding_dim if self.judge_memory else None
            },
            'agents': {
                'count': len(self.agents),
                'stats': {a.name: a.get_stats() for a in self.agents}
            },
            'judge': self.judge.get_stats(),
            'god_protocol': self.god.get_stats(),
            'arena': {
                'total_combats': self.arena.total_combats,
                'current_champion': self.arena.champion.name if self.arena.champion else None
            },
            'history': {
                'total_queries': len(self.query_history),
                'memory_growth_entries': len(self.memory_growth_log),
                'arena_performance_entries': len(self.arena_performance_log)
            }
        }
    
    # --- Private Methods ---
    
    def _log_memory_state(self, event: str) -> None:
        """Log the current memory state for visualization."""
        active_markers = 0
        try:
            active_markers = self.studsar.studsar_network.count_active_markers()
        except Exception:
            active_markers = 0
        self.memory_growth_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'total_markers': self.studsar.studsar_network.get_total_markers(),
            'active_markers': active_markers,
            'entry_number': len(self.memory_growth_log)
        })
    
    def _log_arena_performance(self, result: SMCAResult) -> None:
        """Log arena performance for visualization."""
        combat = result.combat_result
        if not combat:
            return
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': result.query[:50],
            'champion': result.champion_name,
            'champion_score': combat.final_score,
            'total_rounds': combat.total_rounds,
            'total_time': result.total_processing_time,
            'judge_confidence': result.judge_confidence,
            'base_judge_confidence': result.base_judge_confidence,
            'champion_resilience_score': result.champion_resilience_score,
            'god_consulted': result.god_was_consulted,
            'standards_used': result.standards_evolution,
            'agent_scores': {},
            'query_number': len(self.arena_performance_log) + 1
        }
        
        # Extract per-agent scores across rounds
        for agent in self.agents:
            stats = agent.get_stats()
            entry['agent_scores'][agent.name] = {
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['win_rate'],
                'avg_score': stats['average_score']
            }
        
        self.arena_performance_log.append(entry)
    
    def __repr__(self) -> str:
        return (f"SMCAEngine(agents={len(self.agents)}, "
                f"markers={self.studsar.studsar_network.get_total_markers()}, "
                f"queries={len(self.query_history)})")
