"""
Arena — The Combat Field for SMCA Agents.

The Arena is where agents compete to produce the best response.
Uses a champion-based system: the current champion faces each challenger,
with the winner advancing. When the countdown expires, the current
champion's response is the final answer.
"""

import time
from typing import List, Dict, Optional, Any
from .models import AgentResponse, RoundResult, CombatResult
from .agent import CombatAgent
from .judge import Judge
from .countdown import Countdown
from .god_protocol import GodProtocol


class Arena:
    """The SMCA Combat Arena.
    
    Central mechanism where agents compete for the best response.
    The champion-based system ensures progressive refinement:
    - Champion faces the next challenger
    - Winner advances, loser is eliminated
    - Countdown pressure forces convergence
    """
    
    def __init__(self, agents: List[CombatAgent], judge: Judge,
                 countdown: Optional[Countdown] = None,
                 god_protocol: Optional[GodProtocol] = None):
        """Initialize the Arena.
        
        Args:
            agents: List of CombatAgents competing
            judge: The Judge who arbitrates standards and scores
            countdown: Optional Countdown for time pressure
        """
        self.agents = agents
        self.judge = judge
        self.countdown = countdown
        self.god_protocol = god_protocol
        self.champion: Optional[CombatAgent] = None
        self.champion_response: Optional[AgentResponse] = None
        self.champion_score: float = 0.0
        self.round_history: List[RoundResult] = []
        self.total_combats = 0
        
    def run_round(self, query: str, round_number: int,
                 standards: Optional[List[str]] = None) -> RoundResult:
        """Run a single combat round between all agents.
        
        Each agent generates a response, the Judge scores them all,
        and the winner becomes/remains the champion.
        
        Args:
            query: The query to compete over
            round_number: Current round number
            standards: Optional override standards (otherwise Judge generates)
            
        Returns:
            RoundResult with the outcome
        """
        # Get pressure level from countdown
        pressure = 0.0
        if self.countdown:
            pressure = self.countdown.get_pressure_level()

        if standards is None:
            context = {
                'query': query,
                'pressure_level': pressure,
                'round_number': round_number,
                'agents_count': len(self.agents)
            }
            standards = self.judge.generate_standards(round_number, context=context)
        
        print(f"\n  ═══ ARENA ROUND {round_number} ═══")
        print(f"  Standards: {standards}")
        print(f"  Pressure: {pressure:.1%}")
        print(f"  Competitors: {[a.name for a in self.agents]}")
        
        # All agents generate responses
        responses: List[AgentResponse] = []
        for agent in self.agents:
            response = agent.generate_response(query, standards)
            responses.append(response)
            print(f"  [{agent.name}] Response generated "
                  f"(confidence: {response.confidence:.3f}, "
                  f"time: {response.generation_time:.3f}s)")
        
        # Judge evaluates all responses
        scores = self.judge.evaluate_responses(responses, standards)
        winner_name, winner_score = self.judge.determine_winner(scores, responses)
        god_intervention = False

        if self.judge.needs_god_intervention() and self.god_protocol:
            context = {
                'query': query,
                'round': round_number,
                'scores': dict(scores),
                'responses': list(responses),
                'standards': list(standards),
                'judge_confidence': self.judge.get_confidence()
            }
            decision = self.god_protocol.request_intervention(context)
            chosen_winner = (decision or {}).get('chosen_winner')
            if chosen_winner and chosen_winner in scores:
                winner_name = chosen_winner
                winner_score = scores.get(winner_name, 0.0)
                god_intervention = True

        other_scores = {name: score for name, score in scores.items() if name != winner_name}
        loser_name = min(other_scores, key=other_scores.get) if other_scores else ""
        loser_score = other_scores.get(loser_name, 0.0) if loser_name else 0.0

        winner_agent = next((a for a in self.agents if a.name == winner_name), None)
        winner_response = next((r for r in responses if r.agent_name == winner_name), None)

        if winner_agent:
            self.champion = winner_agent
            self.champion_response = winner_response
            self.champion_score = winner_score

        if god_intervention and winner_agent and winner_response and hasattr(winner_agent, "studsar"):
            try:
                winner_markers = set(winner_response.markers_used)
                for mid in winner_markers:
                    winner_agent.studsar.update_marker_reputation(mid, 0.2)
                    # TMDR: Resurrection — grant decay immunity on God resolution
                    winner_agent.studsar.studsar_network.grant_resurrection(mid)
                
                penalized_markers = set()
                for resp in responses:
                    if resp.agent_name == winner_name:
                        continue
                    for mid in resp.markers_used:
                        if mid not in winner_markers and mid not in penalized_markers:
                            winner_agent.studsar.update_marker_reputation(mid, -0.05)
                            penalized_markers.add(mid)
            except Exception:
                pass
        
        # Record results for all agents
        for agent in self.agents:
            agent_score = scores.get(agent.name, 0.0)
            agent.record_result(
                won=(agent.name == winner_name),
                score=agent_score,
                round_number=round_number,
                standards=standards
            )
        
        # Create round result
        round_result = RoundResult(
            round_number=round_number,
            standards_used=standards,
            winner_name=winner_name,
            winner_score=winner_score,
            loser_name=loser_name,
            loser_score=loser_score,
            all_scores=scores,
            all_responses=responses,
            pressure_level=pressure,
            god_intervention=god_intervention
        )
        
        # Record outcome in Judge's memory
        self.judge.record_outcome(round_result)
        
        self.round_history.append(round_result)
        
        print(f"\n  ★ WINNER: {winner_name} ({winner_score:.3f})")
        print(f"  All scores: {', '.join(f'{n}: {s:.3f}' for n, s in scores.items())}")
        print(f"  Judge confidence: {self.judge.get_confidence():.3f}")
        print(f"  ═══════════════════════")
        
        return round_result
    
    def run_combat(self, query: str, max_rounds: int = 5,
                  countdown_seconds: Optional[float] = None,
                  standards_override: Optional[List[List[str]]] = None,
                  selection_mode: str = "champion") -> CombatResult:
        """Run a full combat session with multiple rounds.
        
        Each round uses different standards (emergent from history).
        The countdown pressure reduces available rounds as time runs out.
        When countdown expires, the current champion wins automatically.
        
        Args:
            query: The query to compete over
            max_rounds: Maximum number of rounds (before pressure)
            countdown_seconds: Override countdown duration
            
        Returns:
            CombatResult with the complete combat outcome
        """
        self.total_combats += 1
        start_time = time.time()
        
        # Setup countdown
        if countdown_seconds is not None:
            self.countdown = Countdown(countdown_seconds)
        
        if self.countdown:
            self.countdown.start()
            # Register pressure callbacks
            self.countdown.on_threshold(0.5, lambda t, f, p: 
                print(f"\n  ⚠️  COUNTDOWN: 50% time remaining — pressure rising!"))
            self.countdown.on_threshold(0.25, lambda t, f, p: 
                print(f"\n  🔥 COUNTDOWN: 25% time remaining — accelerating!"))
            self.countdown.on_threshold(0.1, lambda t, f, p: 
                print(f"\n  💀 COUNTDOWN: 10% time remaining — FINAL MOMENTS!"))
        
        print(f"\n╔══════════════════════════════════════════╗")
        print(f"║         SMCA ARENA COMBAT                ║")
        print(f"║  Query: {query[:35]:35s}  ║")
        print(f"║  Agents: {len(self.agents)}, Max Rounds: {max_rounds:3d}         ║")
        if self.countdown:
            print(f"║  Countdown: {self.countdown.total_seconds:.0f}s                         ║")
        print(f"╚══════════════════════════════════════════╝")
        
        # Reset champion for this combat
        self.champion = None
        self.champion_response = None
        self.champion_score = 0.0
        round_results: List[RoundResult] = []
        countdown_expired = False
        god_interventions = 0
        cumulative_scores: Dict[str, float] = {a.name: 0.0 for a in self.agents}
        
        for round_num in range(1, max_rounds + 1):
            # Check countdown
            if self.countdown:
                if self.countdown.is_expired():
                    print(f"\n  ⏰ COUNTDOWN EXPIRED — Champion {self.champion.name if self.champion else 'N/A'} wins!")
                    countdown_expired = True
                    break
                
                # Adjust available rounds based on pressure
                adjusted_max = self.countdown.get_max_rounds(max_rounds)
                if round_num > adjusted_max:
                    print(f"\n  ⏰ Pressure limit reached — no more rounds allowed")
                    countdown_expired = True
                    break
            
            # Run the round
            round_standards = None
            if standards_override and len(standards_override) >= round_num:
                round_standards = standards_override[round_num - 1]
            round_result = self.run_round(query, round_num, standards=round_standards)
            round_results.append(round_result)
            if round_result.god_intervention:
                god_interventions += 1
            for name, score in round_result.all_scores.items():
                cumulative_scores[name] = cumulative_scores.get(name, 0.0) + float(score)
        
        # Stop countdown
        if self.countdown:
            self.countdown.stop()
        
        total_time = time.time() - start_time

        if selection_mode == "cumulative" and round_results:
            best_name = max(cumulative_scores, key=cumulative_scores.get) if cumulative_scores else None
            if best_name:
                best_agent = next((a for a in self.agents if a.name == best_name), None)
                best_response = None
                for rr in reversed(round_results):
                    for resp in rr.all_responses:
                        if resp.agent_name == best_name:
                            best_response = resp
                            break
                    if best_response:
                        break
                if best_agent:
                    self.champion = best_agent
                    self.champion_response = best_response
                    self.champion_score = float(cumulative_scores.get(best_name, 0.0)) / max(len(round_results), 1)
        
        # Build combat result
        result = CombatResult(
            query=query,
            champion_name=self.champion.name if self.champion else "Unknown",
            champion_response=self.champion_response,
            final_score=self.champion_score,
            rounds=round_results,
            total_rounds=len(round_results),
            total_time=total_time,
            countdown_expired=countdown_expired,
            god_interventions=god_interventions,
            metadata={
                'agent_stats': {a.name: a.get_stats() for a in self.agents},
                'judge_stats': self.judge.get_stats(),
                'countdown_status': self.countdown.get_status() if self.countdown else None
            }
        )
        
        print(f"\n╔══════════════════════════════════════════╗")
        print(f"║         COMBAT COMPLETE                  ║")
        print(f"║  Champion: {result.champion_name:30s} ║")
        print(f"║  Score: {result.final_score:.4f}                          ║")
        print(f"║  Rounds: {result.total_rounds}                              ║")
        print(f"║  Time: {result.total_time:.2f}s                           ║")
        print(f"║  Countdown expired: {str(result.countdown_expired):20s}  ║")
        print(f"╚══════════════════════════════════════════╝")
        
        return result
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current agent leaderboard."""
        leaderboard = []
        for agent in self.agents:
            stats = agent.get_stats()
            leaderboard.append(stats)
        
        leaderboard.sort(key=lambda x: x['win_rate'], reverse=True)
        return leaderboard
    
    def reset(self) -> None:
        """Reset the arena for a new combat."""
        self.champion = None
        self.champion_response = None
        self.champion_score = 0.0
        self.round_history = []
    
    def __repr__(self) -> str:
        return (f"Arena(agents={len(self.agents)}, "
                f"champion={self.champion.name if self.champion else 'None'}, "
                f"rounds={len(self.round_history)})")
