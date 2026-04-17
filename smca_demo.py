"""
SMCA Demo — Stud Memory Combat Agents
Full demonstration of Fase 1: Arena Base

This script:
1. Initializes the SMCA Engine with StudSar as the core pivot
2. Ingests documents into shared memory
3. Runs multiple queries through the arena
4. Generates growth charts and dashboard
5. Saves all state to disk
"""

import sys
import os
import argparse
import json
import re
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _interactive_god_callback(context: dict) -> dict:
    scores = context.get('scores', {}) or {}
    standards = context.get('standards', []) or []
    judge_conf = context.get('judge_confidence', 0.0) or 0.0
    print("\n" + "=" * 60)
    print("GOD PROTOCOL — HUMAN ESCALATION")
    print("=" * 60)
    print(f"Standards: {standards}")
    print(f"Judge confidence: {judge_conf:.3f}")
    print("Scores:")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"- {name}: {score:.4f}")

    winner_default = max(scores, key=scores.get) if scores else ""
    raw = input(f"Choose winner (default: {winner_default}): ").strip()
    chosen = raw or winner_default
    chosen_score = float(scores.get(chosen, 0.0)) if scores else 0.0
    return {
        'chosen_winner': chosen,
        'chosen_score': chosen_score,
        'reason': 'human_decision',
    }

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) and gold else 0.0


def _f1(pred: str, gold: str) -> float:
    pred_toks = _normalize_text(pred).split()
    gold_toks = _normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def _contains_answer(pred: str, gold: str) -> bool:
    p = _normalize_text(pred)
    g = _normalize_text(gold)
    if not p or not g:
        return False
    return g in p


def _load_jsonl_dataset(path: str) -> list[dict]:
    items: list[dict] = []
    raw = open(path, "r", encoding="utf-8").read()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _extract_docs(item: dict) -> list[str]:
    if "documents" in item and isinstance(item["documents"], list):
        return [str(x) for x in item["documents"] if str(x).strip()]
    if "contexts" in item and isinstance(item["contexts"], list):
        return [str(x) for x in item["contexts"] if str(x).strip()]
    if "context" in item and isinstance(item["context"], str):
        return [item["context"]]
    return []


def _run_benchmark(args) -> str:
    from src.arena.smca_engine import SMCAEngine
    from src.managers.manager import StudSarManager

    dataset = _load_jsonl_dataset(args.benchmark_dataset)
    shared = StudSarManager()

    seed_text = (
        "Neural associative memory systems retrieve items by comparing vector representations; cosine similarity is a common metric. "
        "Reputation-based boosting can adjust retrieval ranking by upweighting markers that were historically useful. "
        "In transformer attention, attention scores are computed from query-key dot products and normalized with softmax to form weights. "
        "Positional encoding injects token position information so self-attention can represent sequence order. "
        "An LSTM is a Long Short-Term Memory network, a recurrent architecture designed to mitigate vanishing gradients. "
        "The vanishing gradient problem occurs when gradients shrink during backpropagation through many time steps, limiting long-range learning. "
        "Gradient descent is an optimization method that updates parameters in the direction opposite the loss gradient. "
        "In genetic algorithms, selection pressure favors high-fitness candidates; crossover recombines parent chromosomes; mutation introduces random variation."
    )

    fixed_standards = [
        ["precision", "relevance"],
        ["coherence", "depth"],
        ["efficiency", "speed"],
    ]

    baselines = []
    if args.run_baselines:
        baselines = ["full", "full_ziora", "single_agent", "fixed_standards", "no_judge_memory", "no_memory", "cumulative"]
    else:
        baselines = [args.baseline]

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset_path": args.benchmark_dataset,
        "num_samples": len(dataset),
        "baselines": {},
    }

    for baseline in baselines:
        per_sample = []
        evaluated_samples = 0

        enable_judge_memory = baseline not in ["no_judge_memory"]
        allow_empty_memory = baseline in ["no_memory"]
        selection_mode = "cumulative" if baseline == "cumulative" else "champion"
        standards_override = None
        max_rounds = args.max_rounds
        num_agents = args.agents
        enable_red_agent = baseline == "full_ziora"

        if baseline == "single_agent":
            num_agents = 1
            max_rounds = 1
        if baseline == "fixed_standards":
            standards_override = fixed_standards[:max_rounds]
            if len(standards_override) < max_rounds:
                standards_override = standards_override + [fixed_standards[-1]] * (max_rounds - len(standards_override))

        shared_engine = SMCAEngine(
            studsar_manager=None,
            num_agents=num_agents,
            countdown_seconds=args.countdown,
            judge_confidence_threshold=args.judge_threshold,
            auto_god=not args.human_god,
            embedding_generator=shared.embedding_generator,
            device=shared.device,
            enable_judge_memory=enable_judge_memory,
            allow_empty_memory=allow_empty_memory,
            enable_red_agent=enable_red_agent,
            red_alpha=args.alpha,
            red_tau=args.tau,
            red_write_negative_markers=False,
        )
        if args.human_god:
            shared_engine.god.set_human_callback(_interactive_god_callback)

        if baseline != "no_memory":
            shared_engine.ingest_document(seed_text, emotion="informative", source_name="benchmark_seed")
        for item in dataset:
            question = str(item.get("question") or item.get("query") or "").strip()
            answer = str(item.get("answer") or item.get("gold") or item.get("ground_truth") or "").strip()
            metric = str(item.get("metric") or "").strip().lower()
            gt_summary = str(item.get("ground_truth_summary") or "").strip()
            if not question:
                continue

            docs = _extract_docs(item)
            if docs and baseline != "no_memory":
                engine = SMCAEngine(
                    studsar_manager=None,
                    num_agents=num_agents,
                    countdown_seconds=args.countdown,
                    judge_confidence_threshold=args.judge_threshold,
                    auto_god=not args.human_god,
                    embedding_generator=shared.embedding_generator,
                    device=shared.device,
                    enable_judge_memory=enable_judge_memory,
                    allow_empty_memory=allow_empty_memory,
                    enable_red_agent=enable_red_agent,
                    red_alpha=args.alpha,
                    red_tau=args.tau,
                    red_write_negative_markers=False,
                )
                if args.human_god:
                    engine.god.set_human_callback(_interactive_god_callback)
                engine.ingest_document(seed_text, emotion="informative", source_name="benchmark_seed")
                for i, doc in enumerate(docs):
                    engine.ingest_document(doc, emotion="informative", source_name=f"ds_doc_{i}")
            else:
                engine = shared_engine

            res = engine.query(
                question=question,
                max_rounds=max_rounds,
                countdown_seconds=args.countdown,
                standards_override=standards_override,
                selection_mode=selection_mode,
            )

            pred = res.final_answer or ""
            pred_chars = len(pred)
            pred_words = len((pred or "").split())
            markers_accessed = int((res.memory_stats or {}).get("markers_accessed", 0) or 0)
            total_rounds = int((res.combat_result.total_rounds if res.combat_result else 0) or 0)
            god_interventions = int((res.combat_result.god_interventions if res.combat_result else 0) or 0)
            em = 0.0
            f1 = 0.0
            scored = False
            if metric in ["exact_match_f1", "f1", "exact_match"]:
                em = _exact_match(pred, answer)
                f1 = _f1(pred, answer)
                if metric == "exact_match_f1" and answer and _contains_answer(pred, answer):
                    em = 1.0
                    f1 = 1.0
                scored = True
                evaluated_samples += 1
            per_sample.append({
                "id": item.get("id"),
                "question": question,
                "answer": answer,
                "metric": metric,
                "ground_truth_summary": gt_summary,
                "champion": res.champion_name,
                "judge_confidence": res.judge_confidence,
                "base_judge_confidence": getattr(res, "base_judge_confidence", 0.0),
                "champion_resilience_score": getattr(res, "champion_resilience_score", 0.0),
                "god_consulted": res.god_was_consulted,
                "god_interventions": god_interventions,
                "rounds": total_rounds,
                "em": em,
                "f1": f1,
                "time_s": res.total_processing_time,
                "answer_chars": pred_chars,
                "answer_words": pred_words,
                "markers_accessed": markers_accessed,
                "scored": scored,
            })

        scored_only = [x for x in per_sample if x.get("scored")]
        if scored_only:
            avg_em = sum(x["em"] for x in scored_only) / len(scored_only)
            avg_f1 = sum(x["f1"] for x in scored_only) / len(scored_only)
            avg_time = sum(x["time_s"] for x in scored_only) / len(scored_only)
            escalation_rate = sum(1 for x in scored_only if x["god_consulted"]) / len(scored_only)
            avg_conf = sum(x["judge_confidence"] for x in scored_only) / len(scored_only)
            avg_words = sum(x["answer_words"] for x in scored_only) / len(scored_only)
            avg_markers = sum(x["markers_accessed"] for x in scored_only) / len(scored_only)
            avg_rounds = sum(x["rounds"] for x in scored_only) / len(scored_only)
            avg_crs = sum(float(x.get("champion_resilience_score", 0.0) or 0.0) for x in scored_only) / len(scored_only)
        else:
            avg_em = avg_f1 = avg_time = escalation_rate = avg_conf = avg_crs = avg_words = avg_markers = avg_rounds = 0.0

        results["baselines"][baseline] = {
            "aggregate": {
                "avg_em": avg_em,
                "avg_f1": avg_f1,
                "avg_time_s": avg_time,
                "avg_judge_confidence": avg_conf,
                "avg_champion_resilience_score": avg_crs,
                "escalation_rate": escalation_rate,
                "avg_answer_words": avg_words,
                "avg_markers_accessed": avg_markers,
                "avg_rounds": avg_rounds,
                "num_evaluated": len(scored_only),
                "num_total": len(per_sample),
            },
            "samples": per_sample[: args.max_report_samples],
        }

    out_dir = os.path.join(os.path.dirname(__file__), "smca_state")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"benchmark_{results['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nBenchmark saved: {out_path}")
    return out_path


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--countdown", type=float, default=30.0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--judge-threshold", type=float, default=0.7)
    parser.add_argument("--human-god", action="store_true")
    parser.add_argument("--charts-dir", type=str, default="graficagent10")
    parser.add_argument("--red-agent", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--ziora-write-negatives", action="store_true")
    parser.add_argument("--benchmark-dataset", type=str, default="")
    parser.add_argument("--baseline", type=str, default="full",
                        choices=["full", "full_ziora", "single_agent", "fixed_standards", "no_judge_memory", "no_memory", "cumulative"])
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--max-report-samples", type=int, default=25)
    args = parser.parse_args(argv)

    print("\n" + "=" * 60)
    print("SMCA DEMO — Stud Memory Combat Agents")
    print("Arena Cognitiva Evolutiva")
    print("=" * 60 + "\n")

    if args.benchmark_dataset:
        _run_benchmark(args)
        return None


    # 1. INITIALIZE SMCA ENGINE

    from src.arena.smca_engine import SMCAEngine

    engine = SMCAEngine(
        num_agents=args.agents,
        countdown_seconds=args.countdown,
        judge_confidence_threshold=args.judge_threshold,
        auto_god=not args.human_god,
        enable_red_agent=args.red_agent,
        red_alpha=args.alpha,
        red_tau=args.tau,
        red_write_negative_markers=args.ziora_write_negatives
    )
    if args.human_god:
        engine.god.set_human_callback(_interactive_god_callback)


    # 2. INGEST DOCUMENTS INTO STUDSAR

    print("\n" + "=" * 60)
    print("  FASE 1: INGESTING KNOWLEDGE INTO STUDSAR")
    print("=" * 60)

    # Document 1: AI and Machine Learning concepts
    doc_ai = """
    Artificial Intelligence is a broad field of computer science focused on creating 
    intelligent machines that can perform tasks that typically require human intelligence. 
    Machine Learning is a subset of AI that uses statistical techniques to give computer 
    systems the ability to learn from data without being explicitly programmed.

    Deep Learning is a subset of Machine Learning that uses neural networks with many 
    layers (deep neural networks) to model and understand complex patterns in data. 
    Convolutional Neural Networks (CNNs) are particularly effective for image recognition, 
    while Recurrent Neural Networks (RNNs) and Transformers excel at sequential data 
    like natural language.

    Transfer Learning allows models pre-trained on large datasets to be fine-tuned 
    for specific tasks with much less data. This approach has revolutionized NLP with 
    models like BERT, GPT, and T5. Attention mechanisms, introduced in the Transformer 
    architecture, allow models to focus on relevant parts of the input when generating output.

    Reinforcement Learning is another paradigm where agents learn by interacting with 
    an environment and receiving rewards or penalties. This approach has achieved 
    superhuman performance in games like Go and Chess, and is being applied to robotics, 
    autonomous vehicles, and resource optimization.
    """
    engine.ingest_document(doc_ai, emotion='informative', source_name='AI_Fundamentals')

    # Document 2: Memory Systems
    doc_memory = """
    Human memory operates through multiple stages: sensory memory holds information 
    for milliseconds, short-term memory retains data for about 20-30 seconds, and 
    long-term memory can store information indefinitely. The hippocampus plays a crucial 
    role in consolidating short-term memories into long-term storage.

    Associative memory connects related concepts through neural pathways. When one 
    memory is activated, related memories become more accessible through a process 
    called spreading activation. This is why a smell can trigger a vivid childhood memory.

    Memory consolidation occurs primarily during sleep, particularly during REM sleep. 
    During this process, the brain replays and strengthens important memories while 
    pruning less useful ones. This biological mechanism inspired the Dream Mode 
    feature in artificial memory systems.

    Semantic memory stores general knowledge and facts, while episodic memory records 
    personal experiences with temporal context. Both types interact constantly — 
    we use semantic knowledge to interpret new episodes, and episodes can update 
    our semantic understanding.
    """
    engine.ingest_document(doc_memory, emotion='scientific', source_name='Memory_Systems')

    # Document 3: Evolutionary Computation
    doc_evolution = """
    Evolutionary computation is inspired by biological evolution. Genetic algorithms 
    maintain a population of candidate solutions that undergo selection, crossover, 
    and mutation to evolve better solutions over generations. The fitness function 
    determines which individuals survive and reproduce.

    Multi-objective optimization extends evolutionary computation to problems with 
    competing objectives. Pareto-optimal solutions represent the best trade-offs 
    between objectives — no solution can improve one objective without degrading another.

    Competitive coevolution involves multiple populations that evolve against each other, 
    creating an arms race dynamic. Predator-prey systems and game-playing agents 
    are classic examples. This approach can discover strategies that would be impossible 
    to design manually.

    Swarm intelligence, exemplified by ant colony optimization and particle swarm 
    optimization, shows how simple agents following local rules can collectively 
    solve complex problems. Emergence — complex behavior arising from simple interactions — 
    is a key principle in these systems.
    """
    engine.ingest_document(doc_evolution, emotion='inspiring', source_name='Evolutionary_Computing')

    # Add some individual insights
    engine.ingest_text(
        "The combination of associative memory and competitive selection creates "
        "a system that not only remembers but actively curates its knowledge base, "
        "keeping only the most valuable information.",
        emotion='insight'
    )
    engine.ingest_text(
        "Time pressure in decision-making forces convergence toward satisficing "
        "rather than optimizing — finding good-enough solutions quickly rather "
        "than perfect solutions slowly.",
        emotion='strategic'
    )


    # 3. RUN ARENA QUERIES

    print("\n" + "=" * 60)
    print("  FASE 2: ARENA COMBAT — QUERIES")
    print("=" * 60)

    queries = [
        "How does memory consolidation relate to machine learning?",
        "What are the advantages of competitive evolution in AI?",
        "How can attention mechanisms improve memory retrieval?",
        "What parallels exist between human sleep and AI optimization?",
        "How does evolutionary pressure create better solutions?",
    ]

    results = []
    for i, query in enumerate(queries):
        print(f"\n{'#' * 60}")
        print(f"  QUERY {i+1}/{len(queries)}: {query}")
        print(f"{'#' * 60}")

        result = engine.query(
            question=query,
            max_rounds=args.max_rounds,
            countdown_seconds=args.countdown
        )
        results.append(result)


    # 4. GENERATE CHARTS

    print("\n" + "=" * 60)
    print("  FASE 3: GENERATING GROWTH CHARTS")
    print("=" * 60)

    try:
        from src.arena.arena_visualization import generate_all_charts
        chart_dir = os.path.join(os.path.dirname(__file__), args.charts_dir)
        charts = generate_all_charts(engine, output_dir=chart_dir)
        print(f"\n  Generated {len(charts)} visualization charts!")
    except Exception as e:
        print(f"\n  Chart generation error: {e}")
        import traceback
        traceback.print_exc()


    # 5. SAVE STATE

    print("\n" + "=" * 60)
    print("  FASE 4: SAVING COMPLETE STATE")
    print("=" * 60)

    state_dir = os.path.join(os.path.dirname(__file__), "smca_state")
    saved = engine.save_state(directory=state_dir)


    # 6. FINAL SUMMARY

    status = engine.get_system_status()

    print("\n" + "=" * 60)
    print("SMCA DEMO COMPLETE")
    print("=" * 60)
    print(f"StudSar Markers:     {status['studsar']['total_markers']}")
    if 'judge_memory' in status:
        print(f"Judge Markers:      {status['judge_memory']['total_markers']}")
    print(f"Agents:             {status['agents']['count']}")
    print(f"Total Queries:      {status['history']['total_queries']}")
    print(f"Arena Combats:      {status['arena']['total_combats']}")
    print(f"Judge Autonomy:     {status['judge']['autonomy_level']:.1%}")
    print(f"God Interventions:  {status['god_protocol']['total_interventions']}")
    print(f"Current Champion:   {status['arena']['current_champion'] or 'N/A'}")
    print("\nAgent Leaderboard:")

    for agent in engine.agents:
        stats = agent.get_stats()
        print(f"- {agent.name:12s} W:{stats['wins']:2d} L:{stats['losses']:2d} "
              f"Rate:{stats['win_rate']:.0%} Avg:{stats['average_score']:.3f}")

    print("\nFiles saved in:")
    print("- State:  ./smca_state/")
    print(f"- Charts: ./{args.charts_dir}/")
    print("=" * 60 + "\n")

    return engine


if __name__ == "__main__":
    engine = main()
