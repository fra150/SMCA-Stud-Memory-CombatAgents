<div align="center">

<br/>

```
███████╗███╗   ███╗ ██████╗ █████╗ 
██╔════╝████╗ ████║██╔════╝██╔══██╗
███████╗██╔████╔██║██║     ███████║
╚════██║██║╚██╔╝██║██║     ██╔══██║
███████║██║ ╚═╝ ██║╚██████╗██║  ██║
╚══════╝╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═╝
```

# **Stud Memory Combat Agents**
### *Evolutionary Cognitive Arena for Multi-Agent AI Systems*

<br/>

[![GitHub Repo](https://img.shields.io/badge/GitHub-SMCA-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git)
[![Phase](https://img.shields.io/badge/Phase-3%20Complete%20→%204%20Active-blueviolet?style=for-the-badge)](https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-pytest-orange?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

<br/>

> *"SMCA is an evolutionary cognitive arena where StudSar's living memory, agent competition,*  
> *emergent standards, and time pressure converge toward the definitive answer —*  
> *asking God only when the system cannot sustain itself."*

<br/>

</div>

---

## 🧠 What is SMCA?

Current AI systems forget everything between conversations. **SMCA changes that.**

SMCA is not a tool. Not a chatbot. It is a **collective cognitive architecture** for ML teams that combines:

- **Persistent memory** — semantic associations that survive across sessions
- **Anticipatory reasoning** — agents that think ahead before answering
- **Competitive selection** — only the best response wins the arena
- **Minimal human supervision** — the system learns when to ask, and when not to

At its core lives **StudSar** (Neural Associative Memory System) — a memory engine that doesn't retrieve plain text chunks, but builds *living associations* between concepts, enriched with metadata that emulate human memory: emotion, reputation, usage frequency, temporal decay.

---

## 🏛️ Architecture — The Five Pillars

### `01` — StudSar · *The Living Core*

Runs entirely on standard CPU hardware using `sentence-transformers` (all-MiniLM-L6-v2) and advanced text segmentation. Creates intelligent **markers** that retain temporal, emotional, and semantic context — memory that actually *means* something.

### `02` — The Arena · *Combat Between Agents*

No single LLM answers your query. Instead, agents with distinct profiles and specializations — **Precision, Creativity, Synthesis, Speed** — compete head-to-head. The reigning Champion is challenged by the next agent; whoever scores higher takes the throne.

### `03` — The Judge & Emergent Standards · *Dynamic Evaluation*

The combat rules are not fixed. The Judge — a meta-cognitive agent — evaluates responses against **emergent standards** that shift between rounds. Round 1 might weight precision. Round 2, creativity. This ensures no "absolute champion" exists — only the *best archetype for the current need*.

### `04` — The Countdown · *Evolutionary Pressure*

The system operates under **biological urgency**. A dynamic timer progressively reduces available rounds. When time expires, whatever the final Champion holds becomes the definitive answer. Forced convergence under pressure — just like evolution.

### `05` — The God Protocol · *Human Supervision*

*God is the User.* When the Judge cannot confidently declare a winner — due to responses too similar or too contradictory — it triggers an **escalation**. The human steps in. Over time, the Judge memorizes human decisions and becomes increasingly autonomous.

---

## ✅ Current Status — Phase 3 Complete + Phase 4 In Progress

| Component | Status |
|---|---|
| **StudSar Core** — segmentation, live metadata, Dream Mode | ✅ Complete |
| **Arena** — elimination combat & champion management | ✅ Complete |
| **10 Agents + Multi-dimensional Judge** — confidence & emergent standards | ✅ Complete |
| **God Protocol & Countdown** — auto/human escalation + time pressure | ✅ Complete |
| **Ziora (Red Agent Protocol)** — 11th adversarial agent, post-hoc stress test → Champion Resilience Score (CRS) | ✅ Complete |
| **TMDR** — Temporal Memory Decay with Reinforcement Resurrection | ✅ Complete |
| **StudSar-Shadow (anti-poisoning)** — Ziora's negative markers quarantined & promoted only after repeated confirmation | ✅ Complete |
| **Dashboard** — `.png` visualizations with CRS and TMDR Active Markers | ✅ Complete |
| **Phase 4** — Baseline suite, retrieval-hard benchmark, external metrics | 🔄 Active |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Activate your virtual environment and install dependencies
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Full Arena Demo

End-to-end orchestration: loads key ML concepts into StudSar, runs arena combats across multiple agents, prints the God Protocol output and final judgment.

```bash
python smca_demo.py
```

> Output state is saved to `/smca_state/` — visualizations land in `--charts-dir` (default: `graficagent10/`)

---

## 🔧 Usage Examples

```bash
# 10 agents + Ziora (CRS) + charts in agent11/
python smca_demo.py \
  --agents 10 \
  --red-agent \
  --alpha 0.35 \
  --tau 0.08 \
  --charts-dir agent11
```

```bash
# TMDR + Ziora + negative marker write-through
python smca_demo.py \
  --agents 10 \
  --red-agent \
  --alpha 0.35 \
  --tau 0.08 \
  --charts-dir TMDRagent \
  --ziora-write-negatives
```

```bash
# Baseline benchmark suite (pilot set)
python smca_demo.py \
  --benchmark-dataset "Smca baseline eval.json" \
  --run-baselines \
  --agents 10 \
  --alpha 0.35 \
  --tau 0.08
```

```bash
# Retrieval-hard benchmark (answers exist only within ingested documents)
python smca_demo.py \
  --benchmark-dataset "smca_benchmark_retrieval_hard.jsonl" \
  --run-baselines \
  --agents 10 \
  --alpha 0.35 \
  --tau 0.08
```

---

## 🧪 Run Unit Tests

Full coverage across memory architecture and arena logic, built on `pytest`:

```bash
pytest tests/test_arena.py -v --tb=short
```

---

## 🛠️ Phase 4 Tools

Optional diagnostic and calibration scripts:

```bash
# LLM-as-Judge demo (DeepSeek) — uses env variables, no hardcoded keys
python tools/llm_judge_demo.py
```

```bash
# Confidence → accuracy bucket calibration on the latest saved benchmark
python tools/calibrate_judge.py
```

---

## 🔭 Roadmap — Phase 2

| Target | Description |
|---|---|
| **Real API Calls** | Replace simulated reasoning with live LLM API calls over the network |
| **AI-Powered StandardsEngine** | Judge generates emergent standard weights in real-time from historical data |
| **Meta-intelligence** | The system learns to evaluate itself — continuous self-optimization |

---

## 📁 Project Structure

```
SMCA/
├── src/
│   ├── managers/       # StudSar core: segmentation, markers, Dream Mode, TMDR
│   └── arena/          # Arena engine, agent profiles, Judge, Ziora
├── tools/              # LLM judge demo, calibration scripts
├── tests/
│   └── test_arena.py   # Full integration test suite
├── smca_state/         # Persisted memory state
├── smca_demo.py        # Main orchestration script
└── requirements.txt
```

---

## 🔗 Repository

<div align="center">

**Clone it. Run it. Break it. Improve it.**

```bash
git clone https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git
cd SMCA-Stud-Memory-CombatAgents
```

[![View on GitHub](https://img.shields.io/badge/⭐%20Star%20on%20GitHub-SMCA--Stud--Memory--CombatAgents-181717?style=for-the-badge&logo=github)](https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git)

</div>

---

<div align="center">

*Built at the intersection of cognitive science and multi-agent systems.*  
*Memory is not storage — it's the arena where intelligence is forged.*

</div>