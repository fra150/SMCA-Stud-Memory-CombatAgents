"""
BAS Engine — Brain Agent Supreme
Implementation for LOCOMO Benchmark with Post-Retrieval Executor
"""

import time
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from .post_retrieval_executor import PostRetrievalExecutor, execute_numerical_reasoning


@dataclass
class BASResult:
    """Risultato di una query BAS."""
    answer: str
    confidence: float
    participating_agents: List[str]
    rounds_completed: int
    memory_coherence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer': self.answer,
            'confidence': self.confidence,
            'participating_agents': self.participating_agents,
            'rounds_completed': self.rounds_completed,
            'memory_coherence': self.memory_coherence
        }


class SegmentAgent:
    """Agente specializzato su un segmento del documento."""
    
    def __init__(self, agent_id: str, segment_index: int, 
                 segment_text: str, marker_id: Optional[str] = None):
        self.agent_id = agent_id
        self.segment_index = segment_index
        self.segment_text = segment_text
        self.marker_id = marker_id
        self.expertise_score = 1.0
        self.activation_count = 0
        self.last_activation: Optional[datetime] = None
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'segment_index': self.segment_index,
            'segment_text': self.segment_text,
            'marker_id': self.marker_id,
            'expertise_score': self.expertise_score,
            'activation_count': self.activation_count,
            'last_activation': self.last_activation.isoformat() if self.last_activation else None
        }


class BASEngine:
    """BAS Engine with auto-scaling agents and post-retrieval executor."""
    
    def __init__(self, max_agents: int = 200, agents_per_query: int = 10):
        self.max_agents = max_agents
        self.agents_per_query = agents_per_query
        self.segment_agents: Dict[str, SegmentAgent] = {}
        self.query_history: List[BASResult] = []
        self.current_coherence = 0.0
        self.executor = PostRetrievalExecutor()
        
    def ingest_document(self, text: str, source_name: str = "document",
                       segment_size: int = 500) -> int:
        """Ingerisce documento e crea agenti specializzati."""
        # Segmentazione semplice
        segments = []
        words = text.split()
        for i in range(0, len(words), segment_size // 5):  # ~5 parole per token
            segment = ' '.join(words[i:i + segment_size // 5])
            if segment.strip():
                segments.append(segment)
        
        # Crea un agente per segmento
        for idx, segment in enumerate(segments[:self.max_agents]):
            agent_id = f"Agent_{idx}"
            agent = SegmentAgent(
                agent_id=agent_id,
                segment_index=idx,
                segment_text=segment
            )
            self.segment_agents[agent_id] = agent
        
        return len(self.segment_agents)
    
    def _select_agents_for_query(self, query: str, k: Optional[int] = None) -> List[SegmentAgent]:
        """Seleziona agenti rilevanti per la query basandosi sulla similarità semantica."""
        if k is None:
            k = self.agents_per_query
        
        agents = list(self.segment_agents.values())
        
        if not agents:
            return []
        
        # Calcola similarità semantica tra query e ogni segmento
        # Usiamo una semplice metrica di overlap di parole chiave
        query_words = set(query.lower().split())
        
        agent_scores = []
        for agent in agents:
            segment_words = set(agent.segment_text.lower().split())
            
            # Jaccard similarity
            intersection = len(query_words & segment_words)
            union = len(query_words | segment_words)
            similarity = intersection / union if union > 0 else 0.0
            
            # Boost per parole esatte nel segmento
            exact_matches = sum(1 for word in query_words if word in segment_words)
            score = similarity + (exact_matches * 0.1)
            
            agent_scores.append((agent, score))
        
        # Ordina per score decrescente e prendi i top k
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [agent for agent, score in agent_scores[:k]]
        
        return selected
    
    def query(self, question: str, max_rounds: int = 3) -> BASResult:
        """Execute query on document."""
        start_time = time.time()
        
        # Select agents
        selected_agents = self._select_agents_for_query(question)
        
        if not selected_agents:
            return BASResult(
                answer="No agents available",
                confidence=0.0,
                participating_agents=[],
                rounds_completed=0,
                memory_coherence=0.0
            )
        
        # Prepare segments for post-retrieval executor
        segments_data = [
            {'text': agent.segment_text, 'index': agent.segment_index}
            for agent in selected_agents
        ]
        
        # Try post-retrieval executor first (for aggregation queries)
        exec_result = self.executor.execute(question, segments_data)
        
        if exec_result['result'] is not None and exec_result['confidence'] > 0.3:
            # Use executor result for aggregation
            answer = str(exec_result['result'])
            winner_confidence = exec_result['confidence']
            explanation = exec_result.get('explanation', '')
            if explanation:
                answer = f"{answer} ({explanation})"
        else:
            # Fall back to standard retrieval-based answer
            agent_responses = []
            for agent in selected_agents:
                relevance = sum(1 for word in question.lower().split() 
                              if word in agent.segment_text.lower()) / max(1, len(question.split()))
                agent_responses.append((agent, min(1.0, relevance + 0.3)))
            
            winner_agent, winner_confidence = max(agent_responses, key=lambda x: x[1])
            answer = winner_agent.segment_text[:200] + "..." if len(winner_agent.segment_text) > 200 else winner_agent.segment_text
        
        # Calculate memory coherence
        coherence = self._compute_memory_coherence(selected_agents, question)
        self.current_coherence = coherence
        
        result = BASResult(
            answer=answer,
            confidence=winner_confidence,
            participating_agents=[a.agent_id for a in selected_agents],
            rounds_completed=max_rounds,
            memory_coherence=coherence
        )
        
        self.query_history.append(result)
        return result
    
    def _compute_memory_coherence(self, agents: List[SegmentAgent], query: str) -> float:
        """Calcola coerenza memoria tra agenti."""
        if len(agents) < 2:
            return 1.0
        
        # Calcolo semplificato: similarità basata su overlap di parole
        def get_words(text: str) -> set:
            return set(text.lower().split())
        
        agent_word_sets = [get_words(a.segment_text) for a in agents]
        
        # Similarità pairwise media
        similarities = []
        for i in range(len(agent_word_sets)):
            for j in range(i + 1, len(agent_word_sets)):
                intersection = len(agent_word_sets[i] & agent_word_sets[j])
                union = len(agent_word_sets[i] | agent_word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        coherence = sum(similarities) / len(similarities) if similarities else 0.5
        return max(0.0, min(1.0, coherence))
    
    def get_memory_coherence(self) -> float:
        """Ottieni coerenza dall'ultima query."""
        if not self.query_history:
            return 0.0
        return self.query_history[-1].memory_coherence
