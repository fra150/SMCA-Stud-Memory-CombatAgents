"""
BAS Engine — Brain Agent Supreme

Il salto definitivo: un sistema dove il numero di agenti non è fisso ma si
auto-dimensiona in base alla complessità del documento o del problema.

Principio Fondante:
    Documento  10 pagine  →   10 agenti  →  1 cervello
    Documento 100 pagine  →  100 agenti  →  1 cervello
    Documento 500 pagine  →  500 agenti  →  1 cervello

    N pagine = N agenti = N neuroni dello stesso cervello

A differenza di SMCA (agenti fissi), in BAS il documento definisce l'arena.
Il sistema si adatta alla complessità del problema, non il problema al sistema.

Architettura:
                    DIO (utente)
                        ↓
                   [GIUDICE BAS]
                   StudSar vivo
                        ↓
            ┌───────────────────────┐
            │       ARENA BAS       │
            │                       │
            │  [Ag.1]  [Ag.2] ...   │
            │  pag.1   pag.2         │
            │                       │
            │  [Ag.N]  ← N agenti   │
            │  pag.N     = N pagine  │
            └───────────────────────┘
                        ↓
               [STUDSAR CENTRALE]
         memoria condivisa tra tutti gli agenti
                        ↓
              [RISPOSTA SUPREMA]
"""

import time
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from ..managers.manager import StudSarManager
from .agent import CombatAgent
from .arena import Arena
from .judge import Judge
from .countdown import Countdown
from .god_protocol import GodProtocol
from .models import SMCAResult, CombatResult
from .post_retrieval_executor import PostRetrievalExecutor, execute_numerical_reasoning


@dataclass
class SegmentAgent:
    """Un agente dedicato a un singolo segmento/pagina del documento."""

    agent_id: str
    segment_index: int
    segment_text: str
    marker_id: Optional[int]
    expertise_score: float = 1.0
    activation_count: int = 0
    last_activation: Optional[datetime] = None
    combat_agent: Optional[CombatAgent] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'segment_index': self.segment_index,
            'segment_text': self.segment_text[:200] + '...' if len(self.segment_text) > 200 else self.segment_text,
            'marker_id': self.marker_id,
            'expertise_score': self.expertise_score,
            'activation_count': self.activation_count,
            'last_activation': self.last_activation.isoformat() if self.last_activation else None
        }


@dataclass
class BASResult(SMCAResult):
    """Risultato esteso per BAS con metriche specifiche."""

    total_segments: int = 0
    active_agents: int = 0
    segment_distribution: Dict[str, float] = field(default_factory=dict)
    agent_activations: Dict[str, int] = field(default_factory=dict)
    memory_coherence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'total_segments': self.total_segments,
            'active_agents': self.active_agents,
            'segment_distribution': self.segment_distribution,
            'agent_activations': self.agent_activations,
            'memory_coherence_score': self.memory_coherence_score
        })
        return base


class BASEngine:
    """Motore principale di Brain Agent Supreme.

    BAS trasforma radicalmente l'approccio:
    - In SMCA: 10 agenti fissi competono su tutto il documento
    - In BAS: N agenti (uno per segmento) competono con conoscenza specializzata

    Vantaggio devastante:
    Nessun LLM oggi legge davvero 500 pagine. Fingono di farlo — in realtà
    troncano, dimenticano, allucinano. BAS risolve questo problema in modo
    radicalmente diverso:

    LLM tradizionale: legge 500 pagine → dimentica la maggior parte → risponde con lacune
    BAS: 500 agenti, ognuno esperto di 1 pagina, tutti in arena → il campione
         porta la conoscenza più rilevante → risposta suprema senza lacune
    """

    def __init__(self,
                 studsar_manager: Optional[StudSarManager] = None,
                 max_agents: int = 1000,
                 agents_per_query: int = 10,
                 countdown_seconds: float = 60.0,
                 judge_confidence_threshold: float = 0.7,
                 auto_god: bool = True,
                 embedding_generator=None,
                 device=None,
                 enable_judge_memory: bool = True,
                 enable_red_agent: bool = False,
                 red_alpha: float = 0.35,
                 red_tau: float = 0.08,
                 dynamic_agent_selection: bool = True,
                 coherence_weight: float = 0.3):
        """Inizializza il motore BAS.

        Args:
            studsar_manager: Manager StudSar condiviso (sistema nervoso centrale)
            max_agents: Numero massimo di agenti creabili
            agents_per_query: Quanti agenti attivare per query (selezione dinamica)
            countdown_seconds: Durata countdown per combattimento
            judge_confidence_threshold: Soglia confidenza giudice
            auto_god: Se True, protocollo Dio auto-risolve
            dynamic_agent_selection: Se True, seleziona agenti in base alla query
            coherence_weight: Peso per coerenza della memoria nella selezione
        """
        print("\n" + "=" * 70)
        print("  BAS ENGINE — Brain Agent Supreme")
        print("  Initializing scalable neural architecture...")
        print("=" * 70)

        # StudSar come SISTEMA NERVOSO CENTRALE
        # Connette tutti gli agenti in un unico cervello coerente
        self.studsar = studsar_manager if studsar_manager else StudSarManager(
            embedding_generator=embedding_generator,
            device=device
        )

        self.judge_memory = None
        if enable_judge_memory:
            self.judge_memory = StudSarManager(
                embedding_generator=self.studsar.embedding_generator,
                device=self.studsar.device,
                initial_capacity=256
            )

        # Giudice con standard emergenti dalla storia di tutte le arene
        self.judge = Judge(
            studsar_manager=self.judge_memory,
            confidence_threshold=judge_confidence_threshold
        )

        # Post-Retrieval Executor for numerical reasoning
        self.executor = PostRetrievalExecutor()

        # Configurazione agenti
        self.max_agents = max_agents
        self.agents_per_query = agents_per_query
        self.dynamic_agent_selection = dynamic_agent_selection
        self.coherence_weight = coherence_weight

        # Pool di agenti segmentali (creati on-demand)
        self.segment_agents: Dict[str, SegmentAgent] = {}
        self.active_agents_pool: List[CombatAgent] = []

        # Agente rosso opzionale
        self.enable_red_agent = enable_red_agent
        self.red_alpha = red_alpha
        self.red_tau = red_tau
        self.red_agent: Optional[CombatAgent] = None
        if enable_red_agent:
            self.red_agent = CombatAgent(
                name="Ziora",
                strategy="red",
                studsar_manager=self.studsar
            )
            print(f"  ✦ Red Agent enabled: Ziora (adversarial)")

        # Protocollo Dio
        self.god = GodProtocol(self.judge, auto_resolve=auto_god)

        # Arena
        self.countdown_seconds = countdown_seconds
        self.arena: Optional[Arena] = None

        # Statistiche e log
        self.query_history: List[BASResult] = []
        self.memory_growth_log: List[Dict[str, Any]] = []
        self.arena_performance_log: List[Dict[str, Any]] = []

        # Mappatura documenti → agenti
        self.document_segments: Dict[str, List[SegmentAgent]] = {}

        print(f"\n  Sistema configurato:")
        print(f"    StudSar markers: {self.studsar.studsar_network.get_total_markers()}")
        if self.judge_memory:
            print(f"    Judge markers: {self.judge_memory.studsar_network.get_total_markers()}")
        print(f"    Max agents: {max_agents}")
        print(f"    Agents per query: {agents_per_query} ({'dynamic' if dynamic_agent_selection else 'fixed'})")
        print(f"    Countdown: {countdown_seconds}s")
        print(f"    Judge threshold: {judge_confidence_threshold}")
        print(f"    God mode: {'auto' if auto_god else 'human'}")
        print(f"    Coherence weight: {coherence_weight}")
        print("=" * 70 + "\n")

    def ingest_document(self, text: str, source_name: str = "document",
                       emotion: Optional[str] = None,
                       segment_length: int = 100,
                       create_agents: bool = True) -> Dict[str, Any]:
        """Ingerisce un documento e crea agenti dedicati per ogni segmento.

        Questo è il cuore di BAS: ogni segmento genera un agente esperto.

        Args:
            text: Testo del documento da ingerire
            source_name: Identificatore del documento
            emotion: Tag emozionale opzionale
            segment_length: Lunghezza segmenti (parole o frasi)
            create_agents: Se True, crea agenti per ogni segmento

        Returns:
            Statistiche di ingestione
        """
        print(f"\n📥 BAS: Ingesting document '{source_name}' with auto-scaling agents...")

        # Segmenta il documento usando StudSar
        from ..studsar import segment_text

        segments = segment_text(
            text,
            segment_length=segment_length,
            use_spacy=True,
            spacy_sentences_per_segment=3
        )

        if not segments:
            print("  ⚠ No segments generated")
            return {'segments_created': 0, 'agents_created': 0}

        print(f"  Document segmented into {len(segments)} blocks")

        # Verifica limite massimo agenti
        current_agent_count = len(self.segment_agents)
        if current_agent_count + len(segments) > self.max_agents:
            print(f"  ⚠ Warning: Creating {len(segments)} agents would exceed max ({self.max_agents})")
            print(f"  Current agents: {current_agent_count}")

        # Crea marker in StudSar e agenti dedicati
        created_agents = []
        markers_before = self.studsar.studsar_network.get_total_markers()

        for idx, segment in enumerate(segments):
            if not segment.strip():
                continue

            # Genera embedding e aggiungi a StudSar
            embedding = self.studsar.generate_embedding(segment)
            if embedding is None:
                continue

            marker_id = self.studsar.studsar_network.add_marker(segment, embedding)
            if marker_id is None:
                continue

            # Crea agente segmentale dedicato
            if create_agents:
                agent_id = f"{source_name}_seg_{idx}"
                segment_agent = SegmentAgent(
                    agent_id=agent_id,
                    segment_index=idx,
                    segment_text=segment,
                    marker_id=marker_id,
                    expertise_score=1.0,
                    last_activation=datetime.now()
                )

                # Crea CombatAgent wrapper per l'arena
                combat_agent = CombatAgent(
                    name=f"Agent_{idx}",
                    strategy="specialist",  # Strategia specializzata sul segmento
                    studsar_manager=self.studsar
                )
                combat_agent.specialization = {
                    'segment_index': idx,
                    'segment_text': segment,
                    'marker_id': marker_id,
                    'source': source_name
                }
                segment_agent.combat_agent = combat_agent

                self.segment_agents[agent_id] = segment_agent
                created_agents.append(segment_agent)

        # Registra mapping documento → agenti
        self.document_segments[source_name] = created_agents

        # Log crescita memoria
        markers_after = self.studsar.studsar_network.get_total_markers()
        self._log_memory_state(f"ingestion:{source_name}")

        stats = {
            'source': source_name,
            'segments_count': len(segments),
            'agents_created': len(created_agents),
            'markers_before': markers_before,
            'markers_after': markers_after,
            'new_markers': markers_after - markers_before,
            'total_agents': len(self.segment_agents),
            'timestamp': datetime.now().isoformat()
        }

        print(f"  ✓ Created {len(created_agents)} specialized agents")
        print(f"  ✓ Total agents in system: {len(self.segment_agents)}")
        print(f"  ✓ StudSar markers: {markers_after}")

        return stats

    def _select_agents_for_query(self, query: str, k: Optional[int] = None) -> List[SegmentAgent]:
        """Seleziona dinamicamente gli agenti più rilevanti per la query.

        Usa StudSar come sistema nervoso per identificare quali agenti
        (segmenti) sono più pertinenti alla domanda.

        Args:
            query: La domanda da processare
            k: Numero di agenti da selezionare (default: agents_per_query)

        Returns:
            Lista di SegmentAgent selezionati
        """
        if not self.segment_agents:
            print("  ⚠ No agents available")
            return []

        k = k or self.agents_per_query

        if not self.dynamic_agent_selection:
            # Selezione fissa: primi k agenti
            return list(self.segment_agents.values())[:k]

        # Selezione dinamica basata su similarità semantica
        print(f"  🔍 Selecting top {k} agents for query...")

        # Cerca segmenti simili in StudSar
        indices, similarities, segments = self.studsar.search(query, k=min(k * 2, len(self.segment_agents)))

        if not indices:
            # Fallback: selezione casuale ponderata
            import random
            weights = [sa.expertise_score for sa in self.segment_agents.values()]
            selected_ids = random.choices(
                list(self.segment_agents.keys()),
                weights=weights,
                k=min(k, len(self.segment_agents))
            )
            return [self.segment_agents[sid] for sid in selected_ids]

        # Mappa risultati ad agenti
        selected_agents = []
        seen_indices = set()

        for idx, sim in zip(indices, similarities):
            # Trova agente corrispondente a questo segmento
            for agent in self.segment_agents.values():
                if agent.marker_id == idx and agent.agent_id not in [a.agent_id for a in selected_agents]:
                    # Aggiorna expertise score basato sulla similarità
                    agent.expertise_score = max(agent.expertise_score, sim)
                    selected_agents.append(agent)
                    seen_indices.add(idx)

                    if len(selected_agents) >= k:
                        break

            if len(selected_agents) >= k:
                break

        # Se non abbiamo abbastanza agenti, completa con altri
        if len(selected_agents) < k:
            remaining = [a for a in self.segment_agents.values()
                        if a.agent_id not in [sa.agent_id for sa in selected_agents]]
            # Ordina per expertise score
            remaining.sort(key=lambda x: x.expertise_score, reverse=True)
            selected_agents.extend(remaining[:k - len(selected_agents)])

        print(f"  ✓ Selected {len(selected_agents)} agents")

        return selected_agents

    def _build_arena(self, agents: List[SegmentAgent]) -> Arena:
        """Costruisce l'arena con gli agenti selezionati."""

        if not agents:
            # Arena vuota
            self.arena = Arena([], self.judge, god_protocol=self.god)
            return self.arena

        # Estrai CombatAgent dai SegmentAgent
        combat_agents = [sa.combat_agent for sa in agents if sa.combat_agent]

        # Aggiorna pool attivo
        self.active_agents_pool = combat_agents

        # Crea arena
        self.arena = Arena(combat_agents, self.judge, god_protocol=self.god)

        print(f"  Arena built with {len(combat_agents)} agents")

        return self.arena

    def query(self, question: str, max_rounds: int = 3,
             countdown_seconds: Optional[float] = None,
             standards_override: Optional[List[List[str]]] = None,
             selection_mode: str = "champion",
             agents_override: Optional[List[SegmentAgent]] = None) -> BASResult:
        """Processa una query attraverso il sistema BAS completo.

        Pipeline:
        1. StudSar identifica segmenti rilevanti
        2. Agenti dedicati vengono attivati (N agenti = N segmenti rilevanti)
        3. Arena SMCA seleziona il campione
        4. Giudice arbitra con standard emergenti
        5. Countdown forza convergenza
        6. Risposta suprema emerge

        Args:
            question: Domanda da processare
            max_rounds: Round massimi di combattimento
            countdown_seconds: Override durata countdown
            standards_override: Override standard del giudice
            selection_mode: Modalità selezione campione
            agents_override: Override agenti (bypassa selezione dinamica)

        Returns:
            BASResult con risposta e metriche
        """
        start_time = time.time()
        cd_seconds = countdown_seconds or self.countdown_seconds

        print(f"\n{'*' * 70}")
        print(f"  BAS QUERY: {question}")
        print(f"{'*' * 70}")

        # Verifica memoria
        total_markers = self.studsar.studsar_network.get_total_markers()
        if total_markers == 0:
            return BASResult(
                query=question,
                final_answer="[BAS] No memory available. Please ingest documents first.",
                champion_name="None",
                memory_stats={'total_markers': 0},
                total_processing_time=time.time() - start_time
            )

        # Seleziona agenti per questa query
        if agents_override:
            selected_agents = agents_override
        else:
            selected_agents = self._select_agents_for_query(question)

        if not selected_agents:
            return BASResult(
                query=question,
                final_answer="[BAS] No relevant agents found for this query.",
                champion_name="None",
                memory_stats={'total_markers': total_markers},
                total_processing_time=time.time() - start_time
            )

        # Aggiorna statistiche agenti
        now = datetime.now()
        for agent in selected_agents:
            agent.activation_count += 1
            agent.last_activation = now

        # Costruisci arena con agenti selezionati
        self._build_arena(selected_agents)

        # Reset arena per combattimento fresco
        self.arena.reset()

        # Esegui combattimento
        combat_result = self.arena.run_combat(
            query=question,
            max_rounds=max_rounds,
            countdown_seconds=cd_seconds,
            standards_override=standards_override,
            selection_mode=selection_mode
        )

        # Estrai risposta finale
        if combat_result.champion_response:
            final_answer = combat_result.champion_response.text
        else:
            final_answer = "[BAS] Combat completed but no response generated."

        # Calcola resilienza con agente rosso se abilitato
        base_confidence = self.judge.get_confidence()
        resilience_score = 0.0
        final_confidence = base_confidence

        if self.enable_red_agent and self.red_agent and combat_result.champion_response:
            k = min(10, max(total_markers, 0))
            marker_ids, sims, segs = self.studsar.search_with_reputation(
                question, k=k, reputation_weight=1.0
            )
            evidence_segments = list(combat_result.champion_response.marker_segments or [])
            for s in segs or []:
                if s and s not in evidence_segments:
                    evidence_segments.append(s)

            red_response = self.red_agent.generate_response(
                query=question,
                standards=['resilience'],
                context={
                    'champion_text': combat_result.champion_response.text,
                    'evidence_segments': evidence_segments,
                    'tau': self.red_tau,
                    'segments_override': evidence_segments,
                    'similarities_override': list(sims or []),
                    'marker_ids_override': list(marker_ids or []),
                }
            )

            red_metrics = (red_response.metadata or {}).get('red_metrics') or {}
            resilience_score = self.judge.compute_champion_resilience_score(red_metrics)
            final_confidence = self.judge.modulate_final_confidence(
                base_confidence, resilience_score, self.red_alpha
            )

        # Try post-retrieval executor for aggregation queries
        segments_data = [
            {'text': agent.segment_text, 'index': agent.segment_index}
            for agent in selected_agents
        ]
        exec_result = self.executor.execute(question, segments_data)
        if exec_result['result'] is not None and exec_result['confidence'] > 0.3:
            explanation = exec_result.get('explanation', '')
            final_answer = f"{exec_result['result']}"
            if explanation:
                final_answer += f" ({explanation})"

        # Calcola punteggio di coerenza della memoria
        coherence_score = self._compute_memory_coherence(selected_agents, question)

        # Costruisci risultato BAS
        total_time = time.time() - start_time

        # Statistiche distribuzione agenti
        agent_activations = {
            agent.agent_id: agent.activation_count
            for agent in selected_agents
        }

        result = BASResult(
            query=question,
            final_answer=final_answer,
            champion_name=combat_result.champion_name,
            combat_result=combat_result,
            memory_stats={
                'total_markers': total_markers,
                'markers_accessed': sum(
                    len(r.markers_used)
                    for round_r in combat_result.rounds
                    for r in round_r.all_responses
                ),
                'total_agents': len(self.segment_agents),
                'active_agents': len(selected_agents)
            },
            judge_confidence=final_confidence,
            base_judge_confidence=base_confidence,
            champion_resilience_score=resilience_score,
            standards_evolution=[r.standards_used for r in combat_result.rounds],
            god_was_consulted=combat_result.god_interventions > 0,
            total_processing_time=total_time,
            # Metriche BAS-specifiche
            total_segments=len(self.segment_agents),
            active_agents=len(selected_agents),
            segment_distribution=self._compute_segment_distribution(selected_agents),
            agent_activations=agent_activations,
            memory_coherence_score=coherence_score
        )

        # Salva nello storico
        self.query_history.append(result)
        self._log_arena_performance(result)
        self._log_memory_state(f"query:{question[:30]}")

        # Stampa summary
        print(f"\n{'=' * 70}")
        print(f"  BAS ANSWER — RISPOSTA SUPREMA")
        print(f"  Champion: {result.champion_name}")
        print(f"  Active agents: {result.active_agents}/{result.total_segments}")
        print(f"  Confidence: {result.judge_confidence:.3f}")
        print(f"  Memory coherence: {result.memory_coherence_score:.3f}")
        if self.enable_red_agent:
            print(f"  Resilience (CRS): {result.champion_resilience_score:.3f} (alpha={self.red_alpha:.2f})")
        print(f"  Rounds: {combat_result.total_rounds}")
        print(f"  Time: {total_time:.2f}s")
        print(f"{'=' * 70}")
        print(f"\n{final_answer}\n")

        return result

    def _compute_memory_coherence(self, agents: List[SegmentAgent], query: str) -> float:
        """Calcola quanto coerentemente gli agenti rispondono alla query.

        Misura la consistenza semantica tra i segmenti degli agenti attivi.
        Un alto punteggio indica che gli agenti "pensano" in modo coordinato.

        Args:
            agents: Agenti attivi
            query: Domanda originale

        Returns:
            Punteggio di coerenza [0, 1]
        """
        if len(agents) < 2:
            return 1.0  # Singolo agente = massima coerenza

        # Calcola similarità media tra tutti i segmenti degli agenti
        embeddings = []
        for agent in agents:
            emb = self.studsar.generate_embedding(agent.segment_text)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < 2:
            return 0.5

        # Similarità pairwise media
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(embeddings)

        # Prendi la media della triangolare superiore (escludendo diagonale)
        upper_tri_indices = np.triu_indices(len(embeddings), k=1)
        pairwise_sims = similarity_matrix[upper_tri_indices]

        coherence = float(np.mean(pairwise_sims))

        return max(0.0, min(1.0, coherence))

    def _compute_segment_distribution(self, agents: List[SegmentAgent]) -> Dict[str, float]:
        """Calcola la distribuzione dei segmenti per documento sorgente."""
        distribution = {}
        total = len(agents)

        if total == 0:
            return distribution

        for agent in agents:
            # Estrai nome documento dall'agent_id
            parts = agent.agent_id.split('_seg_')
            source = parts[0] if parts else "unknown"
            distribution[source] = distribution.get(source, 0) + 1

        # Normalizza
        for source in distribution:
            distribution[source] = distribution[source] / total

        return distribution

    def _log_memory_state(self, event: str):
        """Registra stato corrente della memoria."""
        log_entry = {
            'event': event,
            'total_markers': self.studsar.studsar_network.get_total_markers(),
            'total_agents': len(self.segment_agents),
            'timestamp': datetime.now().isoformat()
        }
        self.memory_growth_log.append(log_entry)

    def _log_arena_performance(self, result: BASResult):
        """Registra performance dell'arena."""
        log_entry = {
            'query': result.query,
            'champion': result.champion_name,
            'confidence': result.judge_confidence,
            'active_agents': result.active_agents,
            'total_agents': result.total_segments,
            'processing_time': result.total_processing_time,
            'rounds': len(result.standards_evolution),
            'timestamp': datetime.now().isoformat()
        }
        self.arena_performance_log.append(log_entry)

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Restituisce statistiche dettagliate sugli agenti."""
        if not self.segment_agents:
            return {'total_agents': 0}

        activations = [a.activation_count for a in self.segment_agents.values()]
        expertise_scores = [a.expertise_score for a in self.segment_agents.values()]

        # Distribuzione per documento
        doc_distribution = {}
        for agent in self.segment_agents.values():
            parts = agent.agent_id.split('_seg_')
            source = parts[0] if parts else "unknown"
            if source not in doc_distribution:
                doc_distribution[source] = 0
            doc_distribution[source] += 1

        return {
            'total_agents': len(self.segment_agents),
            'avg_activations': sum(activations) / len(activations) if activations else 0,
            'max_activations': max(activations) if activations else 0,
            'min_activations': min(activations) if activations else 0,
            'avg_expertise': sum(expertise_scores) / len(expertise_scores) if expertise_scores else 0,
            'documents': doc_distribution,
            'most_active_agents': sorted(
                [(a.agent_id, a.activation_count) for a in self.segment_agents.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def get_memory_coherence(self) -> float:
        """Ottieni coerenza dall'ultima query."""
        if not self.query_history:
            return 0.0
        return self.query_history[-1].memory_coherence_score

    def save_state(self, directory: str = "bas_state") -> Dict[str, str]:
        """Salva lo stato completo del sistema BAS."""
        os.makedirs(directory, exist_ok=True)
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Salva rete StudSar
        pth_path = os.path.join(directory, f"bas_studsar_network_{timestamp}.pth")
        self.studsar.save(pth_path)
        saved_files['studsar_network'] = pth_path

        if self.judge_memory:
            judge_pth_path = os.path.join(directory, f"bas_judge_studsar_{timestamp}.pth")
            self.judge_memory.save(judge_pth_path)
            saved_files['judge_studsar_network'] = judge_pth_path

        # 2. Salva statistiche agenti
        agents_path = os.path.join(directory, f"bas_agent_stats_{timestamp}.json")
        agent_data = {
            aid: agent.to_dict()
            for aid, agent in self.segment_agents.items()
        }
        with open(agents_path, 'w', encoding='utf-8') as f:
            json.dump(agent_data, f, indent=2, ensure_ascii=False, default=str)
        saved_files['agent_stats'] = agents_path

        # 3. Salva storico query
        history_path = os.path.join(directory, f"bas_query_history_{timestamp}.json")
        history_data = [r.to_dict() for r in self.query_history]
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
        saved_files['query_history'] = history_path

        # 4. Salva log memoria
        growth_path = os.path.join(directory, f"bas_memory_growth_{timestamp}.json")
        with open(growth_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory_growth_log, f, indent=2, ensure_ascii=False, default=str)
        saved_files['memory_growth'] = growth_path

        # 5. Salva performance arena
        perf_path = os.path.join(directory, f"bas_arena_performance_{timestamp}.json")
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(self.arena_performance_log, f, indent=2, ensure_ascii=False, default=str)
        saved_files['arena_performance'] = perf_path

        print(f"\n💾 BAS state saved to {directory}/")
        for component, filepath in saved_files.items():
            print(f"  ✓ {component}: {filepath}")

        return saved_files

    def load_state(self, directory: str,
                   studsar_path: Optional[str] = None,
                   agent_stats_path: Optional[str] = None) -> bool:
        """Carica uno stato BAS precedentemente salvato."""
        print(f"\n📂 Loading BAS state from {directory}/...")

        try:
            # Carica StudSar
            if studsar_path and os.path.exists(studsar_path):
                self.studsar.load(studsar_path)
                print(f"  ✓ Loaded StudSar network from {studsar_path}")

            # Carica statistiche agenti
            if agent_stats_path and os.path.exists(agent_stats_path):
                with open(agent_stats_path, 'r', encoding='utf-8') as f:
                    agent_data = json.load(f)

                # Ricostruisci agenti
                self.segment_agents = {}
                for aid, data in agent_data.items():
                    agent = SegmentAgent(
                        agent_id=data['agent_id'],
                        segment_index=data['segment_index'],
                        segment_text=data['segment_text'],
                        marker_id=data.get('marker_id'),
                        expertise_score=data.get('expertise_score', 1.0),
                        activation_count=data.get('activation_count', 0),
                        last_activation=datetime.fromisoformat(data['last_activation']) if data.get('last_activation') else None
                    )
                    self.segment_agents[aid] = agent

                print(f"  ✓ Loaded {len(self.segment_agents)} agents")

            print(f"  ✓ BAS state restored successfully")
            return True

        except Exception as e:
            print(f"  ✗ Error loading state: {e}")
            return False
