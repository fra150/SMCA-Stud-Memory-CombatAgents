"""
Judge — The Arena's Meta-Cognitive Arbiter.

The Judge is a special agent with its own dedicated StudSar memory.
It arbitrates arena standards, evaluates responses, and knows when
it doesn't know — triggering escalation to God (the human user).
"""

import time
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from .models import AgentResponse, RoundResult
from .standards import StandardsEngine, STANDARD_POOL


class Judge:
    """The Judge: SMCA Arena's meta-cognitive arbiter.
    
    Unlike regular agents, the Judge has:
    - Its own dedicated StudSar for competition history
    - Ability to generate emergent standards from patterns
    - Meta-cognition: knows when it doesn't know (-> God escalation)
    """
    
    # Scoring criteria weights for each standard
    STANDARD_SCORING_RULES = {
        'precision': {
            'weight_similarity': 0.5,
            'weight_confidence': 0.3,
            'weight_markers': 0.2
        },
        'speed': {
            'weight_time': 0.6,
            'weight_confidence': 0.2,
            'weight_markers': 0.2
        },
        'creativity': {
            'weight_diversity': 0.4,
            'weight_length': 0.3,
            'weight_markers': 0.3
        },
        'coherence': {
            'weight_similarity': 0.4,
            'weight_confidence': 0.4,
            'weight_markers': 0.2
        },
        'efficiency': {
            'weight_ratio': 0.5,
            'weight_time': 0.3,
            'weight_confidence': 0.2
        },
        'novelty': {
            'weight_diversity': 0.5,
            'weight_low_sim': 0.3,
            'weight_markers': 0.2
        },
        'depth': {
            'weight_markers': 0.4,
            'weight_length': 0.3,
            'weight_similarity': 0.3
        },
        'relevance': {
            'weight_similarity': 0.6,
            'weight_confidence': 0.3,
            'weight_markers': 0.1
        },
        'conciseness': {
            'weight_ratio': 0.5,
            'weight_confidence': 0.3,
            'weight_brevity': 0.2
        },
        'completeness': {
            'weight_markers': 0.4,
            'weight_length': 0.3,
            'weight_similarity': 0.3
        }
    }
    
    def __init__(self, studsar_manager=None, confidence_threshold: float = 0.7):
        """Initialize the Judge.
        
        Args:
            studsar_manager: Optional dedicated StudSarManager for the Judge's 
                           competition history memory. If None, creates a 
                           lightweight internal memory.
            confidence_threshold: Below this threshold, escalate to God
        """
        self.memory = studsar_manager  # Judge's own StudSar (can be None for lightweight mode)
        self.confidence_threshold = confidence_threshold
        self.competition_history: List[Dict[str, Any]] = []
        self.standards_engine = StandardsEngine(judge_memory=studsar_manager)
        
        # Meta-cognition state
        self._current_confidence = 1.0
        self._ambiguity_log: List[Dict[str, Any]] = []
        self._god_decisions: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_judgments = 0
        self.escalation_count = 0
        
    def generate_standards(self, round_number: int,
                          context: Optional[Dict] = None) -> List[str]:
        """Generate standards for the current round.
        
        Delegates to the StandardsEngine which uses evolutionary
        selection from competition history.
        
        Args:
            round_number: Current round number
            context: Optional context dict
            
        Returns:
            List of standard names for this round
        """
        standards = self.standards_engine.select_standards(round_number, context)
        print(f"  [JUDGE] Round {round_number} standards: {standards}")
        return standards
    
    def evaluate_responses(self, responses: List[AgentResponse],
                          standards: List[str]) -> Dict[str, float]:
        """Evaluate all agent responses against the current standards.
        
        The Judge scores each response independently, then compares.
        If scores are too close, confidence drops (ambiguity).
        
        Args:
            responses: List of all agent responses
            standards: Current round standards
            
        Returns:
            Dict mapping agent_name -> score
        """
        self.total_judgments += 1
        scores = {}
        
        for response in responses:
            score = self._score_response(response, standards)
            scores[response.agent_name] = score
        
        # Calculate confidence based on score dispersion
        if len(scores) >= 2:
            score_values = list(scores.values())
            score_range = max(score_values) - min(score_values)
            score_std = float(np.std(score_values))
            
            # High dispersion = high confidence (clear winner)
            # Low dispersion = low confidence (ambiguous)
            self._current_confidence = min(1.0, score_range * 3 + score_std * 2)
        else:
            self._current_confidence = 0.5

        total_markers_used = sum(len(r.markers_used) for r in responses)
        if total_markers_used == 0:
            self._current_confidence = min(self._current_confidence, 0.05)
        else:
            markers_per_agent = total_markers_used / max(len(responses), 1)
            if markers_per_agent < 1.0:
                self._current_confidence *= markers_per_agent
        
        # Log if ambiguous
        if self._current_confidence < self.confidence_threshold:
            self._ambiguity_log.append({
                'round': len(self.competition_history) + 1,
                'scores': dict(scores),
                'confidence': self._current_confidence,
                'standards': standards,
                'total_markers_used': total_markers_used
            })
        
        return scores
    
    def determine_winner(self, scores: Dict[str, float],
                        responses: List[AgentResponse]) -> Tuple[str, float]:
        """Determine the winner from scored responses.
        
        Args:
            scores: Dict of agent_name -> score
            responses: Original agent responses
            
        Returns:
            Tuple of (winner_name, winner_score)
        """
        if not scores:
            return ("", 0.0)
        
        winner_name = max(scores, key=scores.get)
        winner_score = scores[winner_name]
        
        return winner_name, winner_score
    
    def needs_god_intervention(self) -> bool:
        """Check if the Judge needs human (God) intervention.
        
        This is the meta-cognitive capability: knowing when you don't know.
        
        Returns:
            True if God should be consulted
        """
        return self._current_confidence < self.confidence_threshold
    
    def get_ambiguity_context(self) -> Dict[str, Any]:
        """Get context about why God intervention is needed."""
        if not self._ambiguity_log:
            return {'reason': 'No ambiguity detected'}
        
        latest = self._ambiguity_log[-1]
        return {
            'reason': 'Score ambiguity below confidence threshold',
            'confidence': latest['confidence'],
            'threshold': self.confidence_threshold,
            'conflicting_scores': latest['scores'],
            'standards_used': latest['standards'],
            'suggestion': 'Please review the competing responses and select a winner'
        }
    
    def record_outcome(self, round_result: RoundResult) -> None:
        """Store competition outcome in Judge's memory.
        
        Args:
            round_result: The result of the completed round
        """
        outcome_data = round_result.to_dict()
        self.competition_history.append(outcome_data)
        
        # Record standard performance
        avg_score = sum(round_result.all_scores.values()) / max(len(round_result.all_scores), 1)
        self.standards_engine.record_outcome(
            round_result.round_number,
            round_result.standards_used,
            round_result.winner_score,
            avg_score
        )
        
        # Store in Judge's StudSar memory if available
        if self.memory:
            summary = (f"Round {round_result.round_number}: "
                      f"Winner={round_result.winner_name} (score={round_result.winner_score:.3f}), "
                      f"Standards={round_result.standards_used}, "
                      f"Pressure={round_result.pressure_level:.2f}")
            try:
                self.memory.update_network(summary, emotion='analytical')
            except Exception:
                pass  # Memory storage is optional
    
    def record_god_decision(self, decision: Dict[str, Any]) -> None:
        """Record a decision made by God for future learning.
        
        Over time, the Judge learns from God's decisions and becomes
        more autonomous — needing God less and less.
        
        Args:
            decision: Dict with God's decision details
        """
        self._god_decisions.append(decision)
        self.escalation_count += 1
        
        # Learn from God's decision: adjust confidence threshold
        # As more God decisions are recorded, the Judge becomes wiser
        if len(self._god_decisions) > 5:
            # Gradually lower threshold = Judge becomes more autonomous
            self.confidence_threshold = max(0.3, self.confidence_threshold - 0.02)
        
        # Store in memory if available
        if self.memory and 'reason' in decision:
            learning = (f"GOD_DECISION: {decision.get('reason', 'unknown')} "
                       f"→ Winner: {decision.get('chosen_winner', 'unknown')}")
            try:
                self.memory.update_network(learning, emotion='wisdom')
            except Exception:
                pass
    
    def get_confidence(self) -> float:
        """Get the Judge's current confidence level."""
        return self._current_confidence

    def compute_champion_resilience_score(self, red_metrics: Optional[Dict[str, Any]]) -> float:
        severity = float((red_metrics or {}).get('severity', 1.0))
        severity = min(max(severity, 0.0), 1.0)
        return 1.0 - severity

    def modulate_final_confidence(self, base_confidence: float, resilience_score: float, alpha: float) -> float:
        a = min(max(float(alpha), 0.0), 1.0)
        bc = min(max(float(base_confidence), 0.0), 1.0)
        crs = min(max(float(resilience_score), 0.0), 1.0)
        return (1.0 - a) * bc + a * crs
    
    def get_autonomy_level(self) -> float:
        """Get how autonomous the Judge has become (0.0 to 1.0).
        
        Higher = more autonomous (needs God less).
        """
        if self.total_judgments == 0:
            return 0.0
        
        # Autonomy = 1 - (escalation_rate)
        escalation_rate = self.escalation_count / self.total_judgments
        return 1.0 - escalation_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Judge statistics."""
        return {
            'total_judgments': self.total_judgments,
            'escalation_count': self.escalation_count,
            'autonomy_level': self.get_autonomy_level(),
            'current_confidence': self._current_confidence,
            'confidence_threshold': self.confidence_threshold,
            'god_decisions_count': len(self._god_decisions),
            'ambiguity_events': len(self._ambiguity_log),
            'competition_history_length': len(self.competition_history),
            'standards_evolution': self.standards_engine.get_evolution_data()
        }
    
    # --- Private Methods ---
    
    def _score_response(self, response: AgentResponse,
                       standards: List[str]) -> float:
        """Score a single response against the given standards.
        
        Args:
            response: The agent's response
            standards: Current evaluation standards
            
        Returns:
            Composite score (0.0 to 1.0)
        """
        dimension_scores = self._compute_all_dimensions(response)
        
        # Weight dimensions by standards
        standard_weights = {}
        for standard in standards:
            rules = self.STANDARD_SCORING_RULES.get(standard, {})
            for key, weight in rules.items():
                dim = key.replace('weight_', '')
                standard_weights[dim] = standard_weights.get(dim, 0.0) + weight
        
        # Normalize weights
        total_weight = sum(standard_weights.values())
        if total_weight > 0:
            standard_weights = {k: v / total_weight for k, v in standard_weights.items()}
        
        # Compute weighted score
        total_score = 0.0
        for dim, weight in standard_weights.items():
            total_score += weight * dimension_scores.get(dim, 0.5)
        
        return min(max(total_score, 0.0), 1.0)
    
    def _compute_all_dimensions(self, response: AgentResponse) -> Dict[str, float]:
        """Compute all scoring dimensions for a response."""
        sims = response.similarities or [0.0]
        
        # Similarity: average cosine similarity
        similarity = float(np.mean(sims))
        
        # Confidence: as reported by the agent
        confidence = response.confidence
        
        # Markers: normalized count of markers used
        markers = min(len(response.markers_used) / 5.0, 1.0)
        
        # Time: inverse speed (faster = higher score)
        time_score = 1.0 / (1.0 + response.generation_time) if response.generation_time > 0 else 1.0
        
        # Diversity: standard deviation of similarities (higher = more diverse)
        diversity = float(np.std(sims)) * 3 if len(sims) > 1 else 0.3
        diversity = min(diversity, 1.0)
        
        # Length: normalized response length
        length = min(len(response.text) / 500.0, 1.0)
        
        # Ratio: information density (confidence per unit of text)
        ratio = confidence / max(len(response.text) / 200.0, 0.1) if response.text else 0.0
        ratio = min(ratio, 1.0)
        
        # Low similarity: for novelty (favoring low but non-zero similarity)
        low_sim = 1.0 - similarity if similarity > 0.1 else 0.0
        
        # Brevity: shorter responses score higher
        brevity = 1.0 / (1.0 + len(response.text) / 100.0)
        
        return {
            'similarity': similarity,
            'confidence': confidence,
            'markers': markers,
            'time': time_score,
            'diversity': diversity,
            'length': length,
            'ratio': ratio,
            'low_sim': low_sim,
            'brevity': brevity
        }
    
    def __repr__(self) -> str:
        return (f"Judge(judgments={self.total_judgments}, "
                f"autonomy={self.get_autonomy_level():.2f}, "
                f"confidence={self._current_confidence:.2f})")
