"""
SMCA Arena — Combat Agent System
Stud Memory Combat Agents — Evolutionary Cognitive Arena

The Arena is the central mechanism where agents compete
to produce the best response using StudSar memory.
"""

from .models import AgentResponse, RoundResult, CombatResult, SMCAResult
from .agent import CombatAgent
from .arena import Arena
from .countdown import Countdown
from .judge import Judge
from .standards import StandardsEngine, STANDARD_POOL
from .god_protocol import GodProtocol
from .smca_engine import SMCAEngine

__all__ = [
    'AgentResponse', 'RoundResult', 'CombatResult', 'SMCAResult',
    'CombatAgent', 'Arena', 'Countdown',
    'Judge', 'StandardsEngine', 'STANDARD_POOL',
    'GodProtocol', 'SMCAEngine'
]
