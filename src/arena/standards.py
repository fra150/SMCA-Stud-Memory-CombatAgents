"""
Standards Engine — Emergent Evaluation Criteria for the SMCA Arena.

The most innovative feature of SMCA: evaluation criteria are NOT fixed.
They emerge from the system's history and change every round.

Round 1: standards = [precision, speed]
Round 2: standards = [creativity, coherence]
Round 3: standards = [efficiency, novelty]
    → same document, different results, different champions
"""

import random
import math
from typing import List, Dict, Optional, Any
from collections import Counter


# All possible evaluation standards
STANDARD_POOL = [
    'precision',      # Accuracy and correctness of information
    'speed',          # Response generation time
    'creativity',     # Lateral thinking and novel associations
    'coherence',      # Internal consistency and logical flow
    'efficiency',     # Information density per unit of text
    'novelty',        # Unexpectedness and originality
    'depth',          # Thoroughness of analysis
    'relevance',      # Direct connection to the query
    'conciseness',    # Brevity while maintaining quality
    'completeness'    # Coverage of all aspects
]

# Standards that naturally pair well together
STANDARD_SYNERGIES = {
    'precision': ['relevance', 'coherence'],
    'speed': ['conciseness', 'efficiency'],
    'creativity': ['novelty', 'depth'],
    'coherence': ['precision', 'completeness'],
    'efficiency': ['speed', 'conciseness'],
    'novelty': ['creativity', 'depth'],
    'depth': ['completeness', 'coherence'],
    'relevance': ['precision', 'completeness'],
    'conciseness': ['speed', 'efficiency'],
    'completeness': ['depth', 'relevance']
}

# Standards that create tension (opposites)
STANDARD_TENSIONS = {
    'speed': ['depth', 'completeness'],
    'conciseness': ['depth', 'completeness'],
    'creativity': ['precision', 'coherence'],
    'novelty': ['relevance', 'precision'],
    'depth': ['speed', 'conciseness'],
    'completeness': ['conciseness', 'speed']
}


class StandardsEngine:
    """Generates and evolves evaluation standards for the SMCA Arena.
    
    Standards emerge from competition history — the system learns which
    criteria produce the best answers over time and adapts accordingly.
    """
    
    def __init__(self, judge_memory=None, standards_per_round: int = 2):
        """Initialize the Standards Engine.
        
        Args:
            judge_memory: Optional StudSarManager for the Judge's memory
            standards_per_round: How many standards to select per round
        """
        self.judge_memory = judge_memory
        self.standards_per_round = standards_per_round
        self.history: List[Dict[str, Any]] = []
        self.standard_performance: Dict[str, List[float]] = {s: [] for s in STANDARD_POOL}
        self.round_counter = 0
        
    def select_standards(self, round_number: int, 
                        context: Optional[Dict] = None) -> List[str]:
        """Select standards for the current round.
        
        Uses evolutionary selection: standards that historically produced
        better outcomes are more likely to be selected, but there's always
        a chance of mutation (novel combinations).
        
        Args:
            round_number: Current round number
            context: Optional context (query type, urgency, etc.)
            
        Returns:
            List of selected standard names
        """
        self.round_counter = round_number
        
        if not self.history:
            selected = self._initial_selection(context)
        else:
            selected = self._evolutionary_selection(context)
        
        # Record selection
        self.history.append({
            'round': round_number,
            'standards': selected,
            'method': 'initial' if round_number <= 1 else 'evolutionary',
            'context': context
        })
        
        return selected
    
    def record_outcome(self, round_number: int, standards: List[str],
                      winner_score: float, avg_score: float) -> None:
        """Record how well a set of standards performed.
        
        Args:
            round_number: Round number
            standards: Standards used in this round
            winner_score: Score of the winning agent
            avg_score: Average score across all agents
        """
        # Higher winner score + larger gap from average = better standards
        quality = winner_score * 0.6 + (winner_score - avg_score) * 0.4
        
        for standard in standards:
            if standard in self.standard_performance:
                self.standard_performance[standard].append(quality)
    
    def evolve_from_history(self, results: List[Dict]) -> List[str]:
        """Evolve standards based on accumulated results.
        
        This is the core emergence mechanism: standards that consistently
        produce high-quality winners get higher selection probability.
        
        Args:
            results: List of round result dictionaries
            
        Returns:
            Evolved set of standards
        """
        # Calculate fitness for each standard
        fitness = {}
        for standard, scores in self.standard_performance.items():
            if scores:
                # Weighted average favoring recent results
                weights = [1.0 + 0.1 * i for i in range(len(scores))]
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                total_weight = sum(weights)
                fitness[standard] = weighted_sum / total_weight
            else:
                fitness[standard] = 0.5  # Neutral fitness for untested standards
        
        return self._weighted_selection(fitness)
    
    def get_standard_rankings(self) -> List[Dict[str, Any]]:
        """Get current rankings of all standards by performance."""
        rankings = []
        for standard in STANDARD_POOL:
            scores = self.standard_performance.get(standard, [])
            if scores:
                avg_score = sum(scores) / len(scores)
                trend = scores[-1] - scores[0] if len(scores) > 1 else 0.0
            else:
                avg_score = 0.0
                trend = 0.0
            
            rankings.append({
                'standard': standard,
                'average_score': avg_score,
                'times_used': len(scores),
                'trend': trend,
                'recent_scores': scores[-5:] if scores else []
            })
        
        rankings.sort(key=lambda x: x['average_score'], reverse=True)
        return rankings
    
    def get_evolution_data(self) -> Dict[str, Any]:
        """Get data for visualizing standards evolution."""
        return {
            'history': self.history,
            'performance': {k: list(v) for k, v in self.standard_performance.items()},
            'rankings': self.get_standard_rankings(),
            'total_rounds': self.round_counter
        }
    
    # --- Private Methods ---
    
    def _initial_selection(self, context: Optional[Dict] = None) -> List[str]:
        """Select standards for the first round."""
        if context and 'preferred_standards' in context:
            return context['preferred_standards'][:self.standards_per_round]

        context_standards = self._standards_from_context(context)
        if context_standards:
            return context_standards
        
        # Select a balanced pair: one "quality" and one "efficiency" standard
        quality_standards = ['precision', 'depth', 'completeness', 'coherence']
        efficiency_standards = ['speed', 'conciseness', 'efficiency']
        
        selected = [random.choice(quality_standards)]
        
        if self.standards_per_round > 1:
            remaining_pool = [s for s in STANDARD_POOL if s != selected[0]]
            # Prefer synergistic or tension-creating combinations
            if random.random() < 0.3:  # 30% chance of tension
                tension_options = STANDARD_TENSIONS.get(selected[0], [])
                if tension_options:
                    selected.append(random.choice(tension_options))
                else:
                    selected.append(random.choice(remaining_pool))
            else:
                selected.append(random.choice(remaining_pool))
        
        return selected[:self.standards_per_round]
    
    def _evolutionary_selection(self, context: Optional[Dict] = None) -> List[str]:
        """Select standards using evolutionary pressure from history."""
        # Calculate fitness scores
        fitness = {}
        for standard, scores in self.standard_performance.items():
            if scores:
                # Recent performance weighted more heavily
                recent = scores[-3:] if len(scores) >= 3 else scores
                fitness[standard] = sum(recent) / len(recent)
            else:
                # Untested standards get exploration bonus
                fitness[standard] = 0.6  # Slightly above neutral

        for s in self._standards_from_context(context):
            if s in fitness:
                fitness[s] *= 1.35

        memory_bias = self._standards_from_memory(context)
        for s in memory_bias:
            if s in fitness:
                fitness[s] *= 1.25
        
        pressure = float((context or {}).get('pressure_level', 0.0) or 0.0)
        mutation_prob = 0.05 if pressure >= 0.65 else 0.2
        if random.random() < mutation_prob:
            return random.sample(STANDARD_POOL, min(self.standards_per_round, len(STANDARD_POOL)))
        
        # Avoid repeating exact same standards as last round
        if self.history:
            last_standards = self.history[-1].get('standards', [])
            # Reduce fitness of recently used standards
            for s in last_standards:
                if s in fitness:
                    fitness[s] *= 0.5
        
        return self._weighted_selection(fitness)

    def _standards_from_context(self, context: Optional[Dict]) -> List[str]:
        if not context:
            return []
        candidates: List[str] = []
        query = str(context.get('query', '') or '').strip().lower()
        pressure = float(context.get('pressure_level', 0.0) or 0.0)

        if pressure >= 0.8:
            candidates.extend(['speed', 'conciseness'])
        elif pressure >= 0.55:
            candidates.extend(['efficiency', 'speed'])

        if query:
            if any(k in query for k in ['why', 'how', 'relate', 'relationship', 'parallels', 'because']):
                candidates.extend(['coherence', 'depth'])
            if any(k in query for k in ['compare', 'versus', 'advantages', 'trade-off', 'pros', 'cons']):
                candidates.extend(['completeness', 'precision'])
            if any(k in query for k in ['summarize', 'tl;dr', 'brief']):
                candidates.extend(['conciseness'])
            if any(k in query for k in ['improve', 'optimize', 'best practice', 'practical']):
                candidates.extend(['relevance', 'efficiency'])

        deduped = [s for s in candidates if s in STANDARD_POOL]
        result: List[str] = []
        for s in deduped:
            if s not in result:
                result.append(s)

        if not result:
            return []

        selected = [result[0]]
        if self.standards_per_round > 1:
            synergy = STANDARD_SYNERGIES.get(selected[0], [])
            for s in result[1:]:
                if s in synergy and s not in selected:
                    selected.append(s)
                    break
            if len(selected) < self.standards_per_round:
                remaining = [s for s in result[1:] if s not in selected]
                if remaining:
                    selected.append(remaining[0])

        while len(selected) < self.standards_per_round:
            fallback = random.choice([s for s in STANDARD_POOL if s not in selected])
            selected.append(fallback)

        return selected[:self.standards_per_round]

    def _standards_from_memory(self, context: Optional[Dict]) -> List[str]:
        if not self.judge_memory:
            return []
        query = str((context or {}).get('query', '') or '').strip()
        if not query:
            return []
        try:
            query_embedding = self.judge_memory.generate_embedding(query)
            marker_ids, _, _ = self.judge_memory.studsar_network.search_similar_markers(query_embedding, k=3)
        except Exception:
            return []

        emotion_to_standards = {
            'scientific': ['precision', 'relevance'],
            'informative': ['precision', 'completeness'],
            'analytical': ['coherence', 'precision'],
            'strategic': ['efficiency', 'speed'],
            'insight': ['novelty', 'creativity'],
            'inspiring': ['creativity', 'novelty'],
            'wisdom': ['coherence', 'depth']
        }

        counter: Counter = Counter()
        for mid in marker_ids:
            emotion = self.judge_memory.studsar_network.id_to_emotion.get(mid)
            for s in emotion_to_standards.get(str(emotion or '').lower(), []):
                counter[s] += 1

        ranked = [s for s, _ in counter.most_common() if s in STANDARD_POOL]
        return ranked[: self.standards_per_round]
    
    def _weighted_selection(self, fitness: Dict[str, float]) -> List[str]:
        """Select standards with probability proportional to fitness."""
        standards = list(fitness.keys())
        scores = [max(fitness[s], 0.01) for s in standards]  # Minimum probability
        
        # Softmax-like selection
        total = sum(scores)
        probabilities = [s / total for s in scores]
        
        selected = []
        remaining_standards = list(standards)
        remaining_probs = list(probabilities)
        
        for _ in range(min(self.standards_per_round, len(remaining_standards))):
            # Normalize remaining probabilities
            total_prob = sum(remaining_probs)
            if total_prob <= 0:
                break
            normalized = [p / total_prob for p in remaining_probs]
            
            # Weighted random choice
            r = random.random()
            cumulative = 0.0
            chosen_idx = len(remaining_standards) - 1  # Default to last
            for i, prob in enumerate(normalized):
                cumulative += prob
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            selected.append(remaining_standards[chosen_idx])
            remaining_standards.pop(chosen_idx)
            remaining_probs.pop(chosen_idx)
        
        return selected
