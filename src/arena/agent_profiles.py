"""
Agent Profiles — Predefined specializations for SMCA Combat Agents.

Each agent has a unique character that influences how it retrieves,
processes, and presents information from StudSar memory.

There is no absolute champion — only the contextual champion.
An agent weak in Round 1 can dominate in Round 3 when standards change.
"""

from typing import List, Dict, Any

# The 10 specialized agent profiles
AGENT_PROFILES = [
    {
        'name': 'Precision',
        'strategy': 'precision',
        'description': 'Maximum accuracy, specific citations. Trusts high-similarity markers.',
        'strengths': ['precision', 'relevance', 'coherence'],
        'weaknesses': ['creativity', 'novelty', 'speed']
    },
    {
        'name': 'Creative',
        'strategy': 'creativity',
        'description': 'Lateral associations, original insights. Connects distant concepts.',
        'strengths': ['creativity', 'novelty', 'depth'],
        'weaknesses': ['precision', 'speed', 'conciseness']
    },
    {
        'name': 'Speed',
        'strategy': 'speed',
        'description': 'Rapid response, essential information. Prefers quick retrieval.',
        'strengths': ['speed', 'efficiency', 'conciseness'],
        'weaknesses': ['depth', 'completeness', 'creativity']
    },
    {
        'name': 'Depth',
        'strategy': 'depth',
        'description': 'Deep multi-layered analysis. Retrieves many markers for full context.',
        'strengths': ['depth', 'completeness', 'coherence'],
        'weaknesses': ['speed', 'conciseness', 'efficiency']
    },
    {
        'name': 'Synthesizer',
        'strategy': 'synthesis',
        'description': 'Combines information from multiple sources into unified view.',
        'strengths': ['completeness', 'coherence', 'depth'],
        'weaknesses': ['speed', 'conciseness', 'novelty']
    },
    {
        'name': 'Challenger',
        'strategy': 'contrarian',
        'description': 'Seeks contradictions, presents opposing viewpoints.',
        'strengths': ['novelty', 'creativity', 'depth'],
        'weaknesses': ['precision', 'relevance', 'speed']
    },
    {
        'name': 'Pragmatist',
        'strategy': 'practical',
        'description': 'Focus on applicability and actionable information.',
        'strengths': ['relevance', 'efficiency', 'precision'],
        'weaknesses': ['creativity', 'novelty', 'depth']
    },
    {
        'name': 'Historian',
        'strategy': 'historical',
        'description': 'Temporal context, evolution tracking. Deep memory retrieval.',
        'strengths': ['depth', 'completeness', 'coherence'],
        'weaknesses': ['speed', 'conciseness', 'novelty']
    },
    {
        'name': 'Minimalist',
        'strategy': 'concise',
        'description': 'Distills the essence, removes noise. Quality over quantity.',
        'strengths': ['conciseness', 'efficiency', 'speed'],
        'weaknesses': ['depth', 'completeness', 'creativity']
    },
    {
        'name': 'Explorer',
        'strategy': 'novel',
        'description': 'Seeks unexpected connections, explores knowledge periphery.',
        'strengths': ['novelty', 'creativity', 'depth'],
        'weaknesses': ['precision', 'relevance', 'speed']
    }
]


def get_profiles(num_agents: int = 2) -> List[Dict[str, Any]]:
    """Get agent profiles for the specified number of agents.
    
    Args:
        num_agents: Number of agents to create (1-10)
        
    Returns:
        List of agent profile dictionaries
    """
    num_agents = min(max(num_agents, 1), len(AGENT_PROFILES))
    
    if num_agents == 1:
        return [AGENT_PROFILES[0]]
    if num_agents <= 2:
        # Phase 1: Precision vs Creative (contrasting strategies)
        return [AGENT_PROFILES[0], AGENT_PROFILES[1]]
    elif num_agents <= 4:
        # Phase 2: Add Speed and Depth
        return AGENT_PROFILES[:4]
    else:
        return AGENT_PROFILES[:num_agents]


def get_profile_by_name(name: str) -> Dict[str, Any]:
    """Get a specific profile by name."""
    for profile in AGENT_PROFILES:
        if profile['name'].lower() == name.lower():
            return profile
    return AGENT_PROFILES[0]  # Default to Precision


def get_all_strategies() -> List[str]:
    """Get all available strategy names."""
    return [p['strategy'] for p in AGENT_PROFILES]
