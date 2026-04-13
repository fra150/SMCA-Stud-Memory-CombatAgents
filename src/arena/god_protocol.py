"""
God Protocol — Human Escalation Interface for SMCA.

God is the human user. God does not manage the system — God transcends it.
God intervenes only in edge cases that the Judge cannot resolve autonomously.

Over time, the Judge asks God less and less. StudSar's memory makes
the Judge wiser: it learns from God's resolved cases and incorporates
them into its own decision-making capabilities.

The goal is progressive full autonomy, maintaining a human umbilical 
cord for unsolvable cases.
"""

from typing import Dict, Optional, Any, List, Callable
from datetime import datetime


class GodProtocol:
    """Interface for human escalation in the SMCA system.
    
    When the Judge's confidence is too low, the God Protocol activates
    to request human intervention. Each God decision is recorded and
    used for future learning, progressively reducing the need for
    human intervention.
    """
    
    def __init__(self, judge, auto_resolve: bool = True):
        """Initialize the God Protocol.
        
        Args:
            judge: Reference to the Judge instance
            auto_resolve: If True, auto-resolve without actual human input
                         (for testing/demo). In production, set to False.
        """
        self.judge = judge
        self.auto_resolve = auto_resolve
        self.escalation_history: List[Dict[str, Any]] = []
        self.total_interventions = 0
        self.auto_resolve_count = 0
        
        # Human callback (set externally for real human interaction)
        self._human_callback: Optional[Callable] = None
        
    def set_human_callback(self, callback: Callable) -> None:
        """Set a callback function for actual human interaction.
        
        Args:
            callback: Function that receives context dict and returns decision dict.
                     Signature: callback(context: dict) -> dict
        """
        self._human_callback = callback
        self.auto_resolve = False
    
    def request_intervention(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Request human intervention for an ambiguous situation.
        
        Args:
            context: Dict with details about the ambiguous situation:
                    - 'scores': agent scores
                    - 'responses': agent responses
                    - 'standards': current standards
                    - 'judge_confidence': Judge's confidence level
                    
        Returns:
            Dict with God's decision, or None if not resolved
        """
        self.total_interventions += 1
        
        escalation_record = {
            'timestamp': datetime.now().isoformat(),
            'intervention_number': self.total_interventions,
            'context': {k: v for k, v in context.items() if k != 'responses'},
            'judge_confidence': context.get('judge_confidence', 0.0)
        }
        
        decision = None
        
        if self._human_callback and not self.auto_resolve:
            # Actual human intervention
            try:
                decision = self._human_callback(context)
                escalation_record['method'] = 'human'
                escalation_record['decision'] = decision
            except Exception as e:
                print(f"  [GOD] Human callback error: {e}. Falling back to auto-resolve.")
                decision = self._auto_resolve(context)
                escalation_record['method'] = 'auto_fallback'
                escalation_record['decision'] = decision
        else:
            # Auto-resolve for testing/demo
            decision = self._auto_resolve(context)
            escalation_record['method'] = 'auto'
            escalation_record['decision'] = decision
        
        self.escalation_history.append(escalation_record)
        
        # Record in Judge for learning
        if decision:
            self.judge.record_god_decision(decision)
        
        return decision
    
    def get_autonomy_level(self) -> float:
        """Calculate current autonomy level (0.0 to 1.0).
        
        Autonomy increases as the system makes more autonomous decisions.
        Higher autonomy = God is consulted less frequently.
        """
        return self.judge.get_autonomy_level()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get God Protocol statistics."""
        return {
            'total_interventions': self.total_interventions,
            'auto_resolve_count': self.auto_resolve_count,
            'human_interventions': self.total_interventions - self.auto_resolve_count,
            'autonomy_level': self.get_autonomy_level(),
            'judge_confidence_threshold': self.judge.confidence_threshold,
            'escalation_history_length': len(self.escalation_history),
            'auto_resolve_mode': self.auto_resolve
        }
    
    def get_escalation_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all escalation events."""
        summaries = []
        for record in self.escalation_history:
            summaries.append({
                'number': record['intervention_number'],
                'timestamp': record['timestamp'],
                'method': record['method'],
                'judge_confidence': record.get('judge_confidence', 0.0),
                'chosen_winner': record.get('decision', {}).get('chosen_winner', 'unknown')
            })
        return summaries
    
    # --- Private Methods ---
    
    def _auto_resolve(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-resolve an ambiguous situation.
        
        Uses the highest-scoring agent as the winner. In a real system,
        this would be replaced by actual human input.
        """
        self.auto_resolve_count += 1
        
        scores = context.get('scores', {})
        if scores:
            chosen_winner = max(scores, key=scores.get)
            chosen_score = scores[chosen_winner]
        else:
            chosen_winner = 'unknown'
            chosen_score = 0.0
        
        decision = {
            'chosen_winner': chosen_winner,
            'chosen_score': chosen_score,
            'reason': 'auto_resolved_highest_score',
            'god_confidence': 0.9,  # Auto-resolve is moderately confident
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  [GOD — AUTO] Resolved: {chosen_winner} (score: {chosen_score:.3f})")
        
        return decision
    
    def __repr__(self) -> str:
        return (f"GodProtocol(interventions={self.total_interventions}, "
                f"autonomy={self.get_autonomy_level():.2f})")
