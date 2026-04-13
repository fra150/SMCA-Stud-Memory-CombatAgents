<div align="center">

<br/>

```
██████╗  █████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝
██████╔╝███████║███████╗
██╔══██╗██╔══██║╚════██║
██████╔╝██║  ██║███████║
╚═════╝ ╚═╝  ╚═╝╚══════╝
```

# **Brain Agent Supreme**
### *Dynamic Cognitive Architecture for Long-Context Multi-Agent Systems*

<br/>
<div align="center">
[![GitHub Repo](https://img.shields.io/badge/GitHub-BAS-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git)
[![Phase](https://img.shields.io/badge/Phase-v1.0.0%20Complete-success?style=for-the-badge)](https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git)
[![LOCOMO](https://img.shields.io/badge/LOCOMO-73.3%25%20Accuracy-blue?style=for-the-badge)](locomo_benchmark.py)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Needle](https://img.shields.io/badge/Needle-100%25-success?style=for-the-badge)](locomo_benchmark.py)

<br/>

> *"BAS is the brain to SMCA's organ — where dynamic agent scaling meets post-retrieval numerical reasoning,*  
> *transforming long documents into N specialized agents, each expert on one page,*  
> *connected by StudSar as a central nervous system."*

<br/>

</div>

---

## 🧠 What is BAS?

Current LLMs fail at long context. **BAS changes the paradigm.**

Instead of asking one model to read 500 pages (and inevitably forget), BAS creates **500 specialized agents**, each expert on one page, connected by StudSar as a unified brain. It is a **dynamic cognitive architecture** that combines:

- **Dynamic agent scaling** — N agents = N document segments (not fixed)
- **Semantic selection** — agents chosen by cosine similarity, not position
- **Post-retrieval reasoning** — dedicated executor for numerical aggregation
- **Context-aware filtering** — pre-calculation filtering by transaction type
- **Perfect needle retrieval** — 100% accuracy on facts hidden in 50-segment documents
- **Flawless Multi-Hop Reasoning** — 100% accuracy on correlating independent segments

At its core lives **StudSar Centrale** — the central memory system that transforms static document segments into living associative knowledge, enriched with emotional tags and temporal metadata.

---

## 🏛️ Architecture — The Five Pillars

### `01` — BAS Engine · *The Neural Cortex*

Dynamic agent creation where document complexity dictates agent count. Runs entirely on standard CPU using `sentence-transformers` for embedding generation. Each agent holds exactly one segment, creating true expertise rather than generalized forgetting.

### `02` — Semantic Selection · *The Thalamus*

No positional bias. The system calculates cosine similarity between query embedding and segment embeddings, selecting only the most relevant agents for combat. **Impact:** Needle accuracy improved from 50% → 100%.

### `03` — Post-Retrieval Executor · *The Prefrontal Cortex*

A dedicated numerical reasoning layer that separates retrieval from computation. Features multi-language keyword mapping (Italian "entrate" → English "income"), strict type filtering, and automatic net balance computation (income - expense).

### `04` — StudSar Centrale · *The Hippocampus*

The shared memory system connecting all agents as a unified brain. Agents don't just retrieve text — they access living markers with emotional resonance, usage frequency, and temporal decay. Memory coherence monitoring via TMDR (Temporal Memory Decay Regression).

### `05` — Giudice BAS · *The Executive Control*

Emergent standards evaluation with confidence scoring. The Judge determines when aggregation requires the Post-Retrieval Executor versus direct answer generation. When uncertainty exceeds thresholds, **Dio** (the User) intervenes.

---

## ✅ Current Status — v1.0.0 Complete

| Component | Status | Performance |
|---|---|---|
| **BAS Engine** — Dynamic agent scaling (N segments = N agents) | ✅ Complete | 50 segments tested |
| **Semantic Selection** — Cosine similarity agent ranking | ✅ Complete | 100% Needle accuracy |
| **Post-Retrieval Executor** — Context filtering & net balance | ✅ Complete | 66.7% Aggregation |
| **LOCOMO Benchmark** — Meta AI long-context validation | ✅ Complete | 73.3% overall |
| **TMDR Integration** — Memory coherence monitoring | ✅ Complete | 0.675 coherence score |
| **Multi-language Support** — Italian/English keyword mapping | ✅ Complete | Revenue/Expense detection |
| **Ziora Protocol** — Inter-agent chaining for multi-hop | ✅ Complete | Multi-hop at 100% |

---

## 📊 LOCOMO Benchmark Results

**Final Performance Metrics (Latest Execution):**
```
Total Tests:        30
Correct:            22
Accuracy:           73.3%

Needle (50 seg):    100.0% ✅
Multi-hop:          100.0% ✅
Temporal:           66.7% ✅
Aggregation:        66.7% ✅
Entity Tracking:    33.3% ⚠️
```

| Category | Accuracy | Status | Notes |
|----------|----------|--------|-------|
| **Needle** | 100.0% | ✅ Perfect | Semantic selection eliminates position bias |
| **Multi-hop** | 100.0% | ✅ Perfect | Semantic agent selection enables flawlessly correlating segments |
| **Aggregation** | 66.7% | ✅ Strong | Income/expense filtering + net balance calc |
| **Temporal** | 66.7% | ✅ Pass | Timeline reasoning functional |
| **Entity** | 33.3% | ⚠️ Partial | Tracking entities over changing contexts remains challenging |

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

### Run the BAS Demo

End-to-end orchestration: loads financial documents, creates dynamic agents per segment, executes semantic selection, runs post-retrieval numerical reasoning, and outputs the Risposta Suprema with confidence scores.

```bash
python bas_demo.py
```

> Benchmark results are saved to `locomo_benchmark_results.json`

---

## 🔧 Usage Examples

```bash
# Run full LOCOMO benchmark suite (30 tests)
python locomo_benchmark.py
```

---

## 🔬 Validation & Analysis

Diagnostic metrics from the latest benchmark:
- **Average latency:** 0.35s
- **Average memory coherence:** 0.675

---

## 🔭 Roadmap — Phase 2

| Target | Description |
|---|---|
| **Entity Evolution Tracking** | Improve entity resolving across long timelines/multiple state changes (33% → 70% accuracy) |
| **Scale Testing** | Validate performance at 100+, 500+ segments with coherence monitoring |
| **Cross-Document Reasoning** | Agents that can chain facts across multiple documents |
| **Uncertainty Modeling** | Explicit "unknown" detection when information is unavailable |
| **Real API Integration** | Replace simulated reasoning with live LLM calls for agent responses |

---

## 📁 Project Structure

```
BAS/
├── src/
│   ├── arena/
│   │   ├── bas_engine.py              # Dynamic agent scaling & semantic selection
│   │   └── post_retrieval_executor.py # Numerical reasoning & context filtering
│   └── managers/                      # StudSar Centrale integration
├── tools/                             # Validation & calibration scripts
├── locomo_benchmark.py                # LOCOMO benchmark implementation
├── bas_demo.py                        # Main orchestration script
├── locomo_benchmark_results.json      # Latest benchmark results
└── requirements.txt
```

---

## 🔗 Repository

<div align="center">

**Clone it. Scale it. Benchmark it. Improve it.**

```bash
git clone -b BAS https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git
cd SMCA-Stud-Memory-CombatAgents
```

[![View on GitHub](https://img.shields.io/badge/⭐%20Star%20on%20GitHub-BAS--Brain--Agent--Supreme-181717?style=for-the-badge&logo=github)](https://github.com/fra150/SMCA-Stud-Memory-CombatAgents.git)

</div>

---

<div align="center">

*Three systems. One creator. Inevitable progression.*  
- StudSar → the cell  
- SMCA → the organ  
- **BAS → the brain**

*Memory is not storage — it's the arena where intelligence is forged.*

</div>
