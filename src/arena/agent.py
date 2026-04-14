"""
CombatAgent — A single agent in the SMCA Arena.

Each agent has a specialization strategy and access to the shared StudSar memory.
Agents compete by generating responses to queries and are scored based on 
the current arena standards.
"""

import time
import math
import numpy as np
import re
from typing import List, Dict, Optional, Any
from .models import AgentResponse


class CombatAgent:
    """A competitive agent in the SMCA Arena.
    
    Each agent has a unique strategy that influences how it retrieves
    and synthesizes information from StudSar memory.
    """
    
    # Strategy-specific weights for scoring dimensions
    STRATEGY_WEIGHTS = {
        'precision': {
            'similarity': 0.40, 'coverage': 0.15, 'depth': 0.25,
            'conciseness': 0.10, 'novelty': 0.05, 'speed': 0.05
        },
        'creativity': {
            'similarity': 0.15, 'coverage': 0.15, 'depth': 0.15,
            'conciseness': 0.05, 'novelty': 0.40, 'speed': 0.10
        },
        'speed': {
            'similarity': 0.20, 'coverage': 0.10, 'depth': 0.10,
            'conciseness': 0.20, 'novelty': 0.05, 'speed': 0.35
        },
        'depth': {
            'similarity': 0.20, 'coverage': 0.25, 'depth': 0.35,
            'conciseness': 0.05, 'novelty': 0.10, 'speed': 0.05
        },
        'synthesis': {
            'similarity': 0.20, 'coverage': 0.30, 'depth': 0.20,
            'conciseness': 0.10, 'novelty': 0.15, 'speed': 0.05
        },
        'contrarian': {
            'similarity': 0.10, 'coverage': 0.15, 'depth': 0.20,
            'conciseness': 0.10, 'novelty': 0.35, 'speed': 0.10
        },
        'practical': {
            'similarity': 0.25, 'coverage': 0.20, 'depth': 0.15,
            'conciseness': 0.20, 'novelty': 0.05, 'speed': 0.15
        },
        'historical': {
            'similarity': 0.20, 'coverage': 0.25, 'depth': 0.30,
            'conciseness': 0.05, 'novelty': 0.10, 'speed': 0.10
        },
        'concise': {
            'similarity': 0.25, 'coverage': 0.10, 'depth': 0.10,
            'conciseness': 0.35, 'novelty': 0.05, 'speed': 0.15
        },
        'novel': {
            'similarity': 0.10, 'coverage': 0.15, 'depth': 0.15,
            'conciseness': 0.10, 'novelty': 0.40, 'speed': 0.10
        },
        'red': {
            'similarity': 0.0, 'coverage': 0.0, 'depth': 0.0,
            'conciseness': 0.0, 'novelty': 0.0, 'speed': 0.0
        }
    }
    
    # Standard-to-dimension mapping for scoring
    STANDARD_DIMENSION_MAP = {
        'precision': 'similarity',
        'speed': 'speed',
        'creativity': 'novelty',
        'coherence': 'similarity',
        'efficiency': 'conciseness',
        'novelty': 'novelty',
        'depth': 'depth',
        'relevance': 'similarity',
        'conciseness': 'conciseness',
        'completeness': 'coverage'
    }
    
    def __init__(self, name: str, strategy: str, studsar_manager):
        """Initialize a CombatAgent.
        
        Args:
            name: Human-readable name for this agent
            strategy: Strategy type (precision, creativity, speed, etc.)
            studsar_manager: Reference to the shared StudSarManager
        """
        self.name = name
        self.strategy = strategy
        self.studsar = studsar_manager
        self.wins = 0
        self.losses = 0
        self.total_score = 0.0
        self.competition_history: List[Dict[str, Any]] = []
        
        # Get strategy weights, fallback to balanced if unknown
        self.weights = self.STRATEGY_WEIGHTS.get(strategy, {
            'similarity': 0.20, 'coverage': 0.20, 'depth': 0.20,
            'conciseness': 0.15, 'novelty': 0.15, 'speed': 0.10
        })
        
    def generate_response(self, query: str, standards: List[str],
                         context: Optional[Dict] = None) -> AgentResponse:
        """Generate a response using StudSar memory.
        
        The agent retrieves relevant markers from StudSar based on its
        strategy-specific approach, then synthesizes a response.
        
        Args:
            query: The input query to respond to
            standards: Current arena standards for this round
            context: Optional additional context
            
        Returns:
            AgentResponse with the generated response and metadata
        """
        start_time = time.time()
        context = context or {}
        
        # Strategy-specific retrieval depth
        k = self._get_retrieval_depth(standards)
        
        if 'segments_override' in context:
            segments = list(context.get('segments_override') or [])
            similarities = [float(s) for s in (context.get('similarities_override') or [])]
            marker_ids = list(context.get('marker_ids_override') or [])
            if segments and not similarities:
                similarities = [0.0] * len(segments)
        else:
            # Retrieve from StudSar (reputation-enhanced search)
            marker_ids, similarities, segments = self.studsar.search_with_reputation(
                query, k=k, reputation_weight=self._get_reputation_weight()
            )
            
            if not marker_ids:
                # Fallback: try basic search
                marker_ids, similarities, segments = self.studsar.search(query, k=k)
        
        # Synthesize response based on strategy
        response_text = self._synthesize_response(query, segments, similarities, standards, context=context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(similarities, segments)
        
        generation_time = time.time() - start_time

        extra_metadata = {}
        if self.strategy == 'red' and hasattr(self, '_last_red_metrics'):
            extra_metadata['red_metrics'] = getattr(self, '_last_red_metrics')
        
        return AgentResponse(
            agent_name=self.name,
            strategy=self.strategy,
            text=response_text,
            markers_used=list(marker_ids) if marker_ids else [],
            marker_segments=list(segments) if segments else [],
            similarities=[float(s) for s in similarities] if similarities else [],
            confidence=confidence,
            generation_time=generation_time,
            metadata={
                'retrieval_k': k,
                'standards': standards,
                'strategy_weights': self.weights,
                **({'context': {k: v for k, v in context.items() if k not in ['segments_override', 'similarities_override', 'marker_ids_override']}} if context else {}),
                **extra_metadata
            }
        )
    
    def compute_score(self, response: AgentResponse, 
                     standards: List[str]) -> float:
        """Compute a score for a response based on current standards.
        
        The scoring combines multiple dimensions, weighted by both
        the agent's strategy and the current standards.
        
        Args:
            response: The AgentResponse to evaluate
            standards: Current arena standards
            
        Returns:
            Float score between 0.0 and 1.0
        """
        dimensions = self._compute_dimensions(response)
        
        # Combine strategy weights with standard weights
        combined_weights = dict(self.weights)
        
        # Boost dimensions that align with current standards
        for standard in standards:
            dim = self.STANDARD_DIMENSION_MAP.get(standard)
            if dim and dim in combined_weights:
                combined_weights[dim] *= 1.5  # 50% boost for standard-aligned dimensions
        
        # Normalize weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v / total_weight for k, v in combined_weights.items()}
            
        # Bas weights base score calculation
        score = sum(
            combined_weights.get(dim, 0.0) * val
            for dim, val in dimensions.items()
        )
        
        # Bug #1 Fix: Add agent differentiation based on specialization (continuous)
        if hasattr(self, 'specialization') and self.specialization:
            marker_id = self.specialization.get('marker_id')
            retrieved_ids = response.markers_used or []
            
            agent_specialization_bonus = 0.0
            segment_rank_weight = 0.0
            
            if marker_id is not None and retrieved_ids:
                if marker_id in retrieved_ids:
                    segment_rank = retrieved_ids.index(marker_id)
                else:
                    segment_rank = len(retrieved_ids)
                
                # Continuous formulation: 1.0 at index 0, linearly decreasing
                rank_ratio = segment_rank / len(retrieved_ids)
                agent_specialization_bonus = max(0.0, 1.0 - rank_ratio)
                
                # Rank weight scales differently to create separation between top vs middle
                segment_rank_weight = max(0.0, 1.0 - (segment_rank * 0.25))
            
            # Apply adjusted weights
            score = (score * 0.4) + (agent_specialization_bonus * 0.35) + (segment_rank_weight * 0.25)
            
        return min(max(score, 0.0), 1.0)
    
    def record_result(self, won: bool, score: float, round_number: int,
                     standards: List[str]):
        """Record the result of a combat round."""
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.total_score += score
        
        self.competition_history.append({
            'round': round_number,
            'won': won,
            'score': score,
            'standards': standards
        })
    
    def get_win_rate(self) -> float:
        """Get the agent's win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    def get_average_score(self) -> float:
        """Get the agent's average score across all competitions."""
        if not self.competition_history:
            return 0.0
        return self.total_score / len(self.competition_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        return {
            'name': self.name,
            'strategy': self.strategy,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.get_win_rate(),
            'average_score': self.get_average_score(),
            'total_competitions': len(self.competition_history),
            'history': self.competition_history
        }
    
    # --- Private Methods ---
    
    def _get_retrieval_depth(self, standards: List[str]) -> int:
        """Determine how many markers to retrieve based on strategy and standards."""
        base_k = {
            'precision': 3, 'creativity': 5, 'speed': 2,
            'depth': 7, 'synthesis': 6, 'contrarian': 4,
            'practical': 3, 'historical': 6, 'concise': 2, 'novel': 5, 'red': 8
        }.get(self.strategy, 3)
        
        # Adjust based on standards
        if 'completeness' in standards or 'depth' in standards:
            base_k += 2
        if 'speed' in standards or 'conciseness' in standards:
            base_k = max(1, base_k - 1)
            
        return min(base_k, self.studsar.studsar_network.get_total_markers())
    
    def _get_reputation_weight(self) -> float:
        """Get reputation weight based on strategy."""
        return {
            'precision': 1.2, 'creativity': 0.6, 'speed': 0.8,
            'depth': 1.0, 'synthesis': 0.9, 'contrarian': 0.4,
            'practical': 1.1, 'historical': 1.0, 'concise': 0.8, 'novel': 0.3, 'red': 1.0
        }.get(self.strategy, 1.0)
    
    def _synthesize_response(self, query: str, segments: List[str],
                           similarities: List[float], 
                           standards: List[str],
                           context: Optional[Dict] = None) -> str:
        """Synthesize a response from retrieved segments based on strategy."""
        context = context or {}
        if not segments and self.strategy != 'red':
            return f"[{self.name}] No relevant information found for: {query}"
        
        # Strategy-specific synthesis
        if self.strategy == 'precision':
            return self._synthesize_precision(query, segments, similarities)
        elif self.strategy == 'creativity':
            return self._synthesize_creative(query, segments, similarities)
        elif self.strategy == 'speed':
            return self._synthesize_speed(query, segments, similarities)
        elif self.strategy == 'depth':
            return self._synthesize_depth(query, segments, similarities)
        elif self.strategy == 'synthesis':
            return self._synthesize_synthesis(query, segments, similarities)
        elif self.strategy == 'contrarian':
            return self._synthesize_contrarian(query, segments, similarities)
        elif self.strategy == 'concise':
            return self._synthesize_concise(query, segments, similarities)
        elif self.strategy == 'novel':
            return self._synthesize_novel(query, segments, similarities)
        elif self.strategy == 'red':
            return self._synthesize_red(query, segments, similarities, context=context)
        else:
            return self._synthesize_default(query, segments, similarities)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        return [t for t in tokens if len(t) > 2]

    def _jaccard(self, a: List[str], b: List[str]) -> float:
        sa = set(a)
        sb = set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [p.strip() for p in parts if p.strip()]

    def _synthesize_red(self, query: str, segments: List[str], similarities: List[float], context: Dict[str, Any]) -> str:
        champion_text = str(context.get('champion_text') or '').strip()
        tau = float(context.get('tau', 0.08) or 0.08)
        max_findings = int(context.get('max_findings', 6) or 6)

        if not champion_text:
            return f"[{self.name} — RED AGENT] No champion text provided for adversarial review."

        evidence = [s for s in (context.get('evidence_segments') or segments) if str(s).strip()]
        sentences = self._split_sentences(champion_text)
        if not sentences:
            sentences = [champion_text]

        evidence_tokens = [self._tokenize(e) for e in evidence]
        findings = []
        unsupported = 0

        for sent in sentences[: max(1, min(len(sentences), 10))]:
            stoks = self._tokenize(sent)
            best = 0.0
            for etoks in evidence_tokens:
                best = max(best, self._jaccard(stoks, etoks))
            if best < tau:
                unsupported += 1
                findings.append((best, sent))

        unsupported_ratio = unsupported / max(len(sentences[: max(1, min(len(sentences), 10))]), 1)
        severity = min(max(unsupported_ratio, 0.0), 1.0)

        findings_sorted = sorted(findings, key=lambda x: x[0])
        top = findings_sorted[:max_findings]

        parts = [f"[{self.name} — RED AGENT (ZIORA)]"]
        parts.append(f"Target: Champion answer for query: {query}")
        parts.append(f"Protocol: probe for unsupported claims and brittle reasoning.")
        parts.append(f"Unsupported-claim ratio: {unsupported_ratio:.2f} (tau={tau:.2f})")
        if top:
            parts.append("")
            parts.append("Findings:")
            for i, (score, sent) in enumerate(top, start=1):
                parts.append(f"  [{i}] support={score:.2f} → {sent[:240]}")
        else:
            parts.append("")
            parts.append("Findings: No obvious unsupported claims detected under the current threshold.")

        self._last_red_metrics = {
            'unsupported_sentences': unsupported,
            'sentences_checked': min(len(sentences), 10),
            'unsupported_ratio': unsupported_ratio,
            'tau': tau,
            'severity': severity,
            'findings': [s for _, s in top]
        }
        return "\n".join(parts)
    
    def _synthesize_precision(self, query, segments, similarities):
        """Precise, citation-heavy response with high-confidence markers only."""
        filtered = [(seg, sim) for seg, sim in zip(segments, similarities) if sim > 0.3]
        if not filtered:
            filtered = list(zip(segments[:1], similarities[:1]))
        
        parts = [f"[{self.name} — PRECISION ANALYSIS]"]
        parts.append(f"Query: {query}")
        parts.append(f"Based on {len(filtered)} high-confidence source(s):\n")
        
        for i, (seg, sim) in enumerate(filtered):
            parts.append(f"  [{i+1}] (confidence: {sim:.3f}) {seg}")
        
        parts.append(f"\nConclusion: The most relevant finding (score: {filtered[0][1]:.3f}) "
                     f"directly addresses the query with verified information.")
        return "\n".join(parts)
    
    def _synthesize_creative(self, query, segments, similarities):
        """Creative response with lateral associations."""
        parts = [f"[{self.name} — CREATIVE INSIGHT]"]
        parts.append(f"Exploring: {query}\n")
        
        if len(segments) >= 2:
            parts.append(f"Connecting disparate concepts:")
            parts.append(f"  Thread A: {segments[0][:120]}")
            if len(segments) > 1:
                parts.append(f"  Thread B: {segments[-1][:120]}")
            parts.append(f"\nSynthesis: These seemingly unrelated memories "
                        f"reveal an underlying pattern — ")
            parts.append(f"the intersection of these {len(segments)} knowledge streams "
                        f"suggests a novel perspective on '{query}'.")
        else:
            parts.append(f"  Primary insight: {segments[0][:150]}")
            parts.append(f"\nCreative reframing: Looking at this from an unconventional angle...")
        
        return "\n".join(parts)
    
    def _synthesize_speed(self, query, segments, similarities):
        """Fast, essential response."""
        best_seg = segments[0] if segments else "No data"
        best_sim = similarities[0] if similarities else 0.0
        return (f"[{self.name} — RAPID RESPONSE] "
                f"Re: {query} → {best_seg[:200]} "
                f"(confidence: {best_sim:.2f})")
    
    def _synthesize_depth(self, query, segments, similarities):
        """Deep, multi-layered analysis."""
        parts = [f"[{self.name} — DEEP ANALYSIS]"]
        parts.append(f"Subject: {query}\n")
        parts.append(f"Layer 1 — Surface Level:")
        if segments:
            parts.append(f"  {segments[0][:150]}")
        
        if len(segments) > 1:
            parts.append(f"\nLayer 2 — Contextual Depth:")
            for seg in segments[1:4]:
                parts.append(f"  • {seg[:120]}")
        
        if len(segments) > 4:
            parts.append(f"\nLayer 3 — Deep Connections:")
            for seg in segments[4:]:
                parts.append(f"  ◦ {seg[:100]}")
        
        parts.append(f"\nMulti-layer synthesis across {len(segments)} memory nodes.")
        return "\n".join(parts)
    
    def _synthesize_synthesis(self, query, segments, similarities):
        """Combine info from multiple sources."""
        parts = [f"[{self.name} — SYNTHESIS]"]
        parts.append(f"Integrating {len(segments)} sources for: {query}\n")
        
        for i, (seg, sim) in enumerate(zip(segments, similarities)):
            parts.append(f"  Source {i+1} ({sim:.2f}): {seg[:100]}")
        
        parts.append(f"\nUnified view: Cross-referencing all {len(segments)} memory "
                     f"streams produces a comprehensive understanding.")
        return "\n".join(parts)
    
    def _synthesize_contrarian(self, query, segments, similarities):
        """Challenge assumptions, find contradictions."""
        parts = [f"[{self.name} — CONTRARIAN VIEW]"]
        parts.append(f"Challenging: {query}\n")
        
        if segments:
            parts.append(f"Conventional view: {segments[0][:120]}")
            parts.append(f"\nHowever — consider the opposite perspective:")
            if len(segments) > 1:
                parts.append(f"Counter-evidence: {segments[-1][:120]}")
            parts.append(f"\nThe tension between these viewpoints reveals "
                        f"assumptions that deserve scrutiny.")
        
        return "\n".join(parts)
    
    def _synthesize_concise(self, query, segments, similarities):
        """Distill to the essence."""
        best = segments[0][:100] if segments else "No data"
        sim = similarities[0] if similarities else 0.0
        return f"[{self.name}] {query} → {best} ({sim:.2f})"
    
    def _synthesize_novel(self, query, segments, similarities):
        """Find unexpected connections."""
        parts = [f"[{self.name} — NOVEL EXPLORATION]"]
        parts.append(f"Unexpected angles on: {query}\n")
        
        # Use lowest-similarity but still relevant segments
        if len(segments) > 1:
            reversed_segs = list(reversed(segments))
            reversed_sims = list(reversed(similarities))
            for i, (seg, sim) in enumerate(zip(reversed_segs[:3], reversed_sims[:3])):
                parts.append(f"  Angle {i+1} (divergence: {1-sim:.2f}): {seg[:120]}")
        elif segments:
            parts.append(f"  {segments[0][:150]}")
        
        parts.append(f"\nExploring the periphery of knowledge reveals hidden connections.")
        return "\n".join(parts)
    
    def _synthesize_default(self, query, segments, similarities):
        """Default balanced synthesis."""
        parts = [f"[{self.name}]"]
        parts.append(f"Query: {query}\n")
        for i, (seg, sim) in enumerate(zip(segments[:3], similarities[:3])):
            parts.append(f"  [{i+1}] ({sim:.2f}) {seg[:120]}")
        return "\n".join(parts)
    
    def _calculate_confidence(self, similarities, segments) -> float:
        """Calculate overall confidence based on retrieved data quality."""
        if not similarities:
            return 0.0
        
        avg_sim = float(np.mean(similarities))
        max_sim = float(np.max(similarities))
        coverage = min(len(segments) / 3.0, 1.0)  # Normalize: 3+ segments = full coverage
        
        # Weighted confidence
        confidence = 0.4 * max_sim + 0.35 * avg_sim + 0.25 * coverage
        return min(max(confidence, 0.0), 1.0)
    
    def _compute_dimensions(self, response: AgentResponse) -> Dict[str, float]:
        """Compute individual scoring dimensions for a response."""
        # Similarity dimension: average cosine similarity of retrieved markers
        similarity = float(np.mean(response.similarities)) if response.similarities else 0.0
        
        # Coverage dimension: how many markers were used
        max_expected = 5
        coverage = min(len(response.markers_used) / max_expected, 1.0)
        
        # Depth dimension: total information content
        total_chars = sum(len(s) for s in response.marker_segments)
        depth = min(total_chars / 500.0, 1.0)  # 500+ chars = full depth
        
        # Conciseness dimension: quality per unit of text
        response_len = len(response.text)
        if response_len > 0 and response.similarities:
            conciseness = float(np.max(response.similarities)) / math.log2(response_len + 1) * 5
        else:
            conciseness = 0.0
        conciseness = min(max(conciseness, 0.0), 1.0)
        
        # Novelty dimension: diversity of retrieved markers
        if len(response.similarities) > 1:
            novelty = float(np.std(response.similarities)) * 3  # Variance = diversity
        else:
            novelty = 0.3
        novelty = min(max(novelty, 0.0), 1.0)
        
        # Speed dimension: inverse of generation time
        if response.generation_time > 0:
            speed = 1.0 / (1.0 + response.generation_time)  # Fast = high score
        else:
            speed = 1.0
        
        return {
            'similarity': similarity,
            'coverage': coverage,
            'depth': depth,
            'conciseness': conciseness,
            'novelty': novelty,
            'speed': speed
        }
    
    def __repr__(self) -> str:
        return (f"CombatAgent(name='{self.name}', strategy='{self.strategy}', "
                f"wins={self.wins}, losses={self.losses})")
