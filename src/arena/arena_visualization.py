"""
Arena Visualization — Growth Charts for SMCA Memory and Arena Performance.

Generates matplotlib charts showing:
1. Memory growth over time (StudSar markers)
2. Arena performance (agent scores, win rates)
3. Standards evolution
4. Judge confidence and autonomy
5. Countdown pressure curves
"""

import os
import json
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


# SMCA Color Palette
COLORS = {
    'bg_dark': '#0a0a1a',
    'bg_panel': '#12122a',
    'accent_blue': '#4fc3f7',
    'accent_cyan': '#00e5ff',
    'accent_purple': '#b388ff',
    'accent_pink': '#ff80ab',
    'accent_green': '#69f0ae',
    'accent_orange': '#ffab40',
    'accent_red': '#ff5252',
    'accent_yellow': '#ffd740',
    'text_primary': '#e0e0e0',
    'text_secondary': '#9e9e9e',
    'grid': '#1a1a3a',
    'agent_colors': [
        '#4fc3f7', '#ff80ab', '#69f0ae', '#ffab40', '#b388ff',
        '#ff5252', '#00e5ff', '#ffd740', '#80cbc4', '#ef5350'
    ]
}


def generate_all_charts(smca_engine, output_dir: str = "smca_charts") -> List[str]:
    """Generate all SMCA visualization charts.
    
    Args:
        smca_engine: The SMCAEngine instance
        output_dir: Directory for output images
        
    Returns:
        List of generated file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot generate charts: matplotlib not available")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Memory Growth Chart
    path = os.path.join(output_dir, f"memory_growth_{timestamp}.png")
    plot_memory_growth(smca_engine.memory_growth_log, path)
    generated.append(path)
    
    # 2. Arena Performance Chart
    if smca_engine.arena_performance_log:
        path = os.path.join(output_dir, f"arena_performance_{timestamp}.png")
        plot_arena_performance(smca_engine.arena_performance_log, smca_engine.agents, path)
        generated.append(path)
    
    # 3. Agent Comparison Radar
    if smca_engine.agents and any(a.competition_history for a in smca_engine.agents):
        path = os.path.join(output_dir, f"agent_comparison_{timestamp}.png")
        plot_agent_comparison(smca_engine.agents, path)
        generated.append(path)
    
    # 4. Standards Evolution
    if smca_engine.judge.standards_engine.history:
        path = os.path.join(output_dir, f"standards_evolution_{timestamp}.png")
        plot_standards_evolution(smca_engine.judge.standards_engine, path)
        generated.append(path)
    
    # 5. Combined Dashboard
    path = os.path.join(output_dir, f"smca_dashboard_{timestamp}.png")
    plot_dashboard(smca_engine, path)
    generated.append(path)
    
    print(f"\n📊 Generated {len(generated)} charts in '{output_dir}/':")
    for p in generated:
        print(f"   {os.path.basename(p)}")
    
    return generated


def plot_memory_growth(growth_log: List[Dict], output_path: str) -> None:
    """Plot StudSar memory growth over time."""
    if not growth_log:
        print("No memory growth data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_panel'])
    
    entries = [e['entry_number'] for e in growth_log]
    markers = [e['total_markers'] for e in growth_log]
    active = [e.get('active_markers', None) for e in growth_log]
    events = [e['event'] for e in growth_log]
    
    # Plot area fill
    ax.fill_between(entries, markers, alpha=0.3, color=COLORS['accent_cyan'])
    ax.plot(entries, markers, color=COLORS['accent_cyan'], linewidth=2.5, 
            marker='o', markersize=4, markerfacecolor=COLORS['accent_blue'])
    
    # Mark ingestion events
    for i, (entry, marker_count, event) in enumerate(zip(entries, markers, events)):
        if 'ingestion' in event:
            ax.axvline(x=entry, color=COLORS['accent_green'], alpha=0.3, linestyle='--')
            ax.annotate(event.split(':')[-1][:15], (entry, marker_count),
                       textcoords="offset points", xytext=(5, 10),
                       fontsize=7, color=COLORS['accent_green'], alpha=0.8)
        elif 'query' in event:
            ax.axvline(x=entry, color=COLORS['accent_purple'], alpha=0.2, linestyle=':')
    
    title = 'StudSar Memory Growth'
    if any(a is not None for a in active):
        title = 'StudSar Memory Growth (TMDR)'
    ax.set_title(title, fontsize=16, fontweight='bold',
                color=COLORS['text_primary'], pad=15)
    ax.set_xlabel('Event', fontsize=11, color=COLORS['text_secondary'])
    ax.set_ylabel('Total Markers', fontsize=11, color=COLORS['text_secondary'])
    ax.tick_params(colors=COLORS['text_secondary'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    
    if any(a is not None for a in active):
        ax.plot(entries, [a if a is not None else 0 for a in active],
                color=COLORS['accent_orange'], linewidth=2.0, linestyle='-.',
                marker='^', markersize=4, label='Active Markers (TMDR)')
        ax.legend(loc='upper left', facecolor=COLORS['bg_panel'],
                  edgecolor=COLORS['grid'], labelcolor=COLORS['text_primary'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), 
                bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  ✓ Memory growth chart saved: {output_path}")


def plot_arena_performance(perf_log: List[Dict], agents: List, output_path: str) -> None:
    """Plot arena performance — agent scores across queries."""
    if not perf_log:
        print("No arena performance data to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    for ax in axes:
        ax.set_facecolor(COLORS['bg_panel'])
    
    query_numbers = [e['query_number'] for e in perf_log]
    
    # --- Top chart: Champion scores over time ---
    ax1 = axes[0]
    champion_scores = [e['champion_score'] for e in perf_log]
    judge_conf = [e['judge_confidence'] for e in perf_log]
    resilience = [e.get('champion_resilience_score') for e in perf_log]
    
    ax1.plot(query_numbers, champion_scores, color=COLORS['accent_cyan'], 
             linewidth=2.5, marker='D', markersize=6, label='Champion Score',
             markerfacecolor=COLORS['accent_green'])
    ax1.plot(query_numbers, judge_conf, color=COLORS['accent_purple'],
             linewidth=1.5, linestyle='--', marker='s', markersize=4, 
             label='Judge Confidence', alpha=0.7)

    if any(r is not None and float(r) > 0.0 for r in resilience):
        ax1.plot(query_numbers, [float(r or 0.0) for r in resilience], color=COLORS['accent_orange'],
                 linewidth=1.5, linestyle='-.', marker='^', markersize=4,
                 label='Champion Resilience (CRS)', alpha=0.85)
    
    # Mark God interventions
    for e in perf_log:
        if e.get('god_consulted'):
            ax1.axvline(x=e['query_number'], color=COLORS['accent_red'], 
                       alpha=0.4, linestyle=':', linewidth=2)
            ax1.annotate('🙏 GOD', (e['query_number'], max(champion_scores)),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, color=COLORS['accent_red'], ha='center')
    
    ax1.set_title('Arena Combat Performance', fontsize=14, fontweight='bold',
                 color=COLORS['text_primary'])
    ax1.set_ylabel('Score', fontsize=11, color=COLORS['text_secondary'])
    ax1.legend(loc='lower right', facecolor=COLORS['bg_panel'], 
              edgecolor=COLORS['grid'], labelcolor=COLORS['text_primary'])
    ax1.tick_params(colors=COLORS['text_secondary'])
    ax1.grid(True, alpha=0.15, color=COLORS['grid'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['grid'])
    ax1.spines['bottom'].set_color(COLORS['grid'])
    
    # --- Bottom chart: Agent win rates ---
    ax2 = axes[1]
    agent_names = list(perf_log[-1]['agent_scores'].keys()) if perf_log else []
    
    for i, agent_name in enumerate(agent_names):
        color = COLORS['agent_colors'][i % len(COLORS['agent_colors'])]
        win_rates = []
        for e in perf_log:
            agent_data = e.get('agent_scores', {}).get(agent_name, {})
            win_rates.append(agent_data.get('win_rate', 0.0))
        
        ax2.plot(query_numbers, win_rates, color=color, linewidth=2,
                marker='o', markersize=5, label=agent_name)
        ax2.fill_between(query_numbers, win_rates, alpha=0.1, color=color)
    
    ax2.set_title('Agent Win Rates Over Time', fontsize=14, fontweight='bold',
                 color=COLORS['text_primary'])
    ax2.set_xlabel('Query Number', fontsize=11, color=COLORS['text_secondary'])
    ax2.set_ylabel('Win Rate', fontsize=11, color=COLORS['text_secondary'])
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right', facecolor=COLORS['bg_panel'],
              edgecolor=COLORS['grid'], labelcolor=COLORS['text_primary'])
    ax2.tick_params(colors=COLORS['text_secondary'])
    ax2.grid(True, alpha=0.15, color=COLORS['grid'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['grid'])
    ax2.spines['bottom'].set_color(COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  ✓ Arena performance chart saved: {output_path}")


def plot_agent_comparison(agents: List, output_path: str) -> None:
    """Plot radar chart comparing agent capabilities."""
    if not agents:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_panel'])
    
    categories = ['Win Rate', 'Avg Score', 'Total Wins', 'Competitions', 'Consistency']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for i, agent in enumerate(agents):
        stats = agent.get_stats()
        total_comps = max(stats['total_competitions'], 1)
        
        values = [
            stats['win_rate'],
            stats['average_score'],
            min(stats['wins'] / max(total_comps, 1), 1.0),
            min(total_comps / 10.0, 1.0),
            1.0 - (np.std([h['score'] for h in stats['history']]) if stats['history'] else 0.5)
        ]
        values += values[:1]
        
        color = COLORS['agent_colors'][i % len(COLORS['agent_colors'])]
        ax.plot(angles, values, 'o-', linewidth=2, label=agent.name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color=COLORS['text_primary'])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], 
                       fontsize=8, color=COLORS['text_secondary'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    
    ax.set_title('Agent Capability Comparison', fontsize=16, fontweight='bold',
                color=COLORS['text_primary'], pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
             facecolor=COLORS['bg_panel'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text_primary'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  ✓ Agent comparison chart saved: {output_path}")


def plot_standards_evolution(standards_engine, output_path: str) -> None:
    """Plot how standards have evolved over rounds."""
    history = standards_engine.history
    if not history:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    for ax in axes:
        ax.set_facecolor(COLORS['bg_panel'])
    
    # Left: Standards usage heatmap
    ax1 = axes[0]
    from .standards import STANDARD_POOL
    
    all_standards = STANDARD_POOL
    rounds = [h['round'] for h in history]
    usage_matrix = np.zeros((len(all_standards), len(rounds)))
    
    for j, h in enumerate(history):
        for standard in h['standards']:
            if standard in all_standards:
                i = all_standards.index(standard)
                usage_matrix[i][j] = 1.0
    
    im = ax1.imshow(usage_matrix, aspect='auto', cmap='YlOrRd', alpha=0.9)
    ax1.set_yticks(range(len(all_standards)))
    ax1.set_yticklabels(all_standards, fontsize=9, color=COLORS['text_primary'])
    ax1.set_xticks(range(len(rounds)))
    ax1.set_xticklabels([str(r) for r in rounds], fontsize=8, color=COLORS['text_secondary'])
    ax1.set_xlabel('Round', fontsize=11, color=COLORS['text_secondary'])
    ax1.set_title('Standards Usage per Round', fontsize=13, fontweight='bold',
                 color=COLORS['text_primary'])
    
    # Right: Standards performance ranking
    ax2 = axes[1]
    rankings = standards_engine.get_standard_rankings()
    
    if rankings:
        names = [r['standard'] for r in rankings]
        scores = [r['average_score'] for r in rankings]
        times_used = [r['times_used'] for r in rankings]
        
        colors = [COLORS['accent_green'] if s > 0.5 else COLORS['accent_orange'] 
                  if s > 0.3 else COLORS['accent_red'] for s in scores]
        
        bars = ax2.barh(range(len(names)), scores, color=colors, alpha=0.8, height=0.6)
        
        # Add usage count annotations
        for i, (score, used) in enumerate(zip(scores, times_used)):
            ax2.text(score + 0.02, i, f'×{used}', va='center',
                    fontsize=9, color=COLORS['text_secondary'])
        
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=10, color=COLORS['text_primary'])
        ax2.set_xlabel('Average Performance Score', fontsize=11, 
                       color=COLORS['text_secondary'])
        ax2.set_title('Standards Performance Ranking', fontsize=13, 
                     fontweight='bold', color=COLORS['text_primary'])
        ax2.tick_params(colors=COLORS['text_secondary'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(COLORS['grid'])
        ax2.spines['bottom'].set_color(COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  ✓ Standards evolution chart saved: {output_path}")


def plot_dashboard(smca_engine, output_path: str) -> None:
    """Generate a comprehensive SMCA dashboard."""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('SMCA — Stud Memory Combat Agents Dashboard', 
                fontsize=20, fontweight='bold', color=COLORS['accent_cyan'],
                y=0.98)
    
    # --- 1. Memory Growth (top-left, spans 2 cols) ---
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(COLORS['bg_panel'])
    
    if smca_engine.memory_growth_log:
        entries = [e['entry_number'] for e in smca_engine.memory_growth_log]
        markers = [e['total_markers'] for e in smca_engine.memory_growth_log]
        ax1.fill_between(entries, markers, alpha=0.3, color=COLORS['accent_cyan'])
        ax1.plot(entries, markers, color=COLORS['accent_cyan'], linewidth=2, marker='o', ms=3)
    
    ax1.set_title('Memory Growth', fontsize=12, fontweight='bold', color=COLORS['text_primary'])
    ax1.set_ylabel('Markers', fontsize=9, color=COLORS['text_secondary'])
    ax1.tick_params(colors=COLORS['text_secondary'], labelsize=8)
    ax1.grid(True, alpha=0.1, color=COLORS['grid'])
    _style_axis(ax1)
    
    # --- 2. System Status (top-right) ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(COLORS['bg_panel'])
    ax2.axis('off')
    
    status = smca_engine.get_system_status()
    last_crs = None
    if getattr(smca_engine, 'arena_performance_log', None):
        last_crs = smca_engine.arena_performance_log[-1].get('champion_resilience_score')
    status_text = [
        f"StudSar Markers: {status['studsar']['total_markers']}",
        f"Judge Markers: {status.get('judge_memory', {}).get('total_markers', 0)}",
        f"Agents: {status['agents']['count']}",
        f"Total Queries: {status['history']['total_queries']}",
        f"Judge Autonomy: {status['judge']['autonomy_level']:.1%}",
        f"Last CRS: {float(last_crs):.3f}" if last_crs is not None else "Last CRS: N/A",
        f"God Interventions: {status['god_protocol']['total_interventions']}",
        f"Arena Combats: {status['arena']['total_combats']}",
        f"Champion: {status['arena']['current_champion'] or 'N/A'}"
    ]
    
    for i, line in enumerate(status_text):
        color = COLORS['accent_green'] if i == 0 else COLORS['text_primary']
        ax2.text(0.1, 0.9 - i * 0.12, line, fontsize=10, color=color,
                transform=ax2.transAxes, fontfamily='monospace')
    
    ax2.set_title('System Status', fontsize=12, fontweight='bold', 
                 color=COLORS['text_primary'])
    
    # --- 3. Agent Win Rates (middle-left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(COLORS['bg_panel'])
    
    agent_names = [a.name for a in smca_engine.agents]
    win_rates = [a.get_win_rate() for a in smca_engine.agents]
    colors = [COLORS['agent_colors'][i % len(COLORS['agent_colors'])] 
              for i in range(len(agent_names))]
    
    bars = ax3.bar(agent_names, win_rates, color=colors, alpha=0.8, width=0.6)
    for bar, rate in zip(bars, win_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.0%}', ha='center', fontsize=9, color=COLORS['text_primary'])
    
    ax3.set_title('Agent Win Rates', fontsize=12, fontweight='bold',
                 color=COLORS['text_primary'])
    ax3.set_ylim(0, 1.15)
    ax3.tick_params(colors=COLORS['text_secondary'], labelsize=8)
    _style_axis(ax3)
    
    # --- 4. Champion Scores (middle-center) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(COLORS['bg_panel'])
    
    if smca_engine.arena_performance_log:
        qnums = [e['query_number'] for e in smca_engine.arena_performance_log]
        scores = [e['champion_score'] for e in smca_engine.arena_performance_log]
        ax4.plot(qnums, scores, color=COLORS['accent_green'], linewidth=2, marker='D', ms=5)
        ax4.fill_between(qnums, scores, alpha=0.2, color=COLORS['accent_green'])
    
    ax4.set_title('Champion Scores', fontsize=12, fontweight='bold',
                 color=COLORS['text_primary'])
    ax4.tick_params(colors=COLORS['text_secondary'], labelsize=8)
    ax4.grid(True, alpha=0.1, color=COLORS['grid'])
    _style_axis(ax4)
    
    # --- 5. Judge Confidence (middle-right) ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(COLORS['bg_panel'])
    
    if smca_engine.arena_performance_log:
        qnums = [e['query_number'] for e in smca_engine.arena_performance_log]
        confs = [e['judge_confidence'] for e in smca_engine.arena_performance_log]
        ax5.plot(qnums, confs, color=COLORS['accent_purple'], linewidth=2, marker='s', ms=4)
        ax5.axhline(y=smca_engine.judge.confidence_threshold, color=COLORS['accent_red'],
                   linestyle='--', alpha=0.5, label='God Threshold')
        ax5.legend(loc='lower right', fontsize=8, facecolor=COLORS['bg_panel'],
                  edgecolor=COLORS['grid'], labelcolor=COLORS['text_primary'])
    
    ax5.set_title('Judge Confidence', fontsize=12, fontweight='bold',
                 color=COLORS['text_primary'])
    ax5.set_ylim(-0.05, 1.05)
    ax5.tick_params(colors=COLORS['text_secondary'], labelsize=8)
    ax5.grid(True, alpha=0.1, color=COLORS['grid'])
    _style_axis(ax5)
    
    # --- 6. Standards Heatmap (bottom, spans all) ---
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(COLORS['bg_panel'])
    
    std_history = smca_engine.judge.standards_engine.history
    if std_history:
        from .standards import STANDARD_POOL
        all_stds = STANDARD_POOL
        rounds = [h['round'] for h in std_history]
        matrix = np.zeros((len(all_stds), len(rounds)))
        for j, h in enumerate(std_history):
            for s in h['standards']:
                if s in all_stds:
                    matrix[all_stds.index(s)][j] = 1.0
        
        ax6.imshow(matrix, aspect='auto', cmap='YlOrRd', alpha=0.85)
        ax6.set_yticks(range(len(all_stds)))
        ax6.set_yticklabels(all_stds, fontsize=8, color=COLORS['text_primary'])
        ax6.set_xticks(range(len(rounds)))
        ax6.set_xticklabels([str(r) for r in rounds], fontsize=7, color=COLORS['text_secondary'])
    
    ax6.set_title('Standards Evolution Across Rounds', fontsize=12, fontweight='bold',
                 color=COLORS['text_primary'])
    ax6.set_xlabel('Round', fontsize=9, color=COLORS['text_secondary'])
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  ✓ Dashboard saved: {output_path}")


def _style_axis(ax):
    """Apply consistent dark styling to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
