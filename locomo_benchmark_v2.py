"""
LOCOMO Benchmark v2 per BAS — Brain Agent Supreme
Scaled benchmark: 500+ procedurally-generated tests with baselines.

Tests:
  1. Needle in a Haystack — Finding specific facts in long documents
  2. Multi-hop Reasoning — Connecting facts across segments
  3. Temporal Reasoning — Understanding timelines
  4. Aggregation — Summing/calculating distributed data
  5. Entity Tracking — Tracking entities across documents

Baselines:
  A. BM25-only retrieval (no agents)
  B. Single-Agent BAS (no Arena, no Ziora)
  C. Full BAS (complete system)
"""

import sys
import os
import time
import json
import random
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.arena.bas_engine import BASEngine, BASResult
from src.managers import StudSarManager


# ═══════════════════════════════════════════════════════════════════
#  DATA GENERATORS — Procedural test data with controlled variation
# ═══════════════════════════════════════════════════════════════════

# Name pools for variety
PERSON_NAMES = [
    "Marco Rossi", "Laura Bianchi", "Giuseppe Verdi", "Anna Ferrari",
    "Luca Moretti", "Sofia Conti", "Andrea Bruno", "Elena Ricci",
    "Davide Romano", "Giulia Marino", "Matteo Greco", "Chiara Lombardi",
    "Alessio Barbieri", "Francesca Gallo", "Simone Serra", "Valentina Costa",
    "Roberto Fontana", "Martina De Luca", "Federico Mancini", "Aurora Colombo"
]

COMPANY_NAMES = [
    "TechCorp Italia", "Innovazione SpA", "DataSolutions Srl", "AIFactory Roma",
    "NeuraTech Milano", "CloudBase Italia", "BioLogics SpA", "CyberNet Srl",
    "QuantumBit Roma", "SmartGrid SpA", "EcoTech Srl", "DigiFlow Italia",
    "RoboSystems SpA", "MedTech Srl", "FinTech Milano", "AgriSmart Italia"
]

CITY_DATA = [
    ("Milano", "Lombardia", 1396000), ("Roma", "Lazio", 2873000),
    ("Napoli", "Campania", 962003), ("Torino", "Piemonte", 870952),
    ("Firenze", "Toscana", 382258), ("Bologna", "Emilia-Romagna", 394463),
    ("Genova", "Liguria", 560688), ("Palermo", "Sicilia", 657561),
    ("Bari", "Puglia", 316140), ("Venezia", "Veneto", 261905)
]

SECTORS = [
    "Software", "Consulenza", "Finanza", "Biotecnologie", "AI",
    "Cybersecurity", "Cloud", "Energia", "Robotica", "Automotive"
]

SECRET_TYPES = [
    ("informatore", "L'informatore chiave è {name}, contattabile tramite email criptata."),
    ("password", "La password master del sistema è: {code}"),
    ("codice", "Il codice di accesso al vault è: {code}"),
    ("appuntamento", "L'appuntamento segreto è fissato per il {date} alle {time} al {place}"),
    ("trasferimento", "Il trasferimento fondi da {amount}€ è stato autorizzato il {date}"),
    ("chiave_api", "La chiave API è: {code}"),
    ("frequenza", "La frequenza radio segreta è: {code} MHz"),
    ("protocollo", "Il protocollo di emergenza è codificato come: {code}"),
    ("contatto", "Il contatto di riferimento è {name}, reperibile al numero {code}"),
    ("indirizzo", "L'indirizzo sicuro è: Via {name} {code}, piano {floor}")
]

FILLER_TEXTS = [
    "Documento amministrativo contenente informazioni di routine su procedure aziendali e protocolli operativi standard.",
    "Analisi di mercato: Il settore mostra crescita costante con tendenze positive nelle metriche chiave.",
    "Report finanziario: Performance positive across all sectors nel periodo di riferimento.",
    "Nota operativa: Procedure standardizzate per tutti i dipartimenti e le unità operative.",
    "Aggiornamento trimestrale: I risultati mostrano un allineamento con gli obiettivi strategici.",
    "Comunicazione interna: Nuove linee guida per la gestione dei processi aziendali.",
    "Bollettino informativo: Riepilogo delle attività svolte nel periodo corrente.",
    "Memorandum organizzativo: Ristrutturazione dei flussi di lavoro dipartimentali.",
    "Circolare tecnica: Aggiornamento dei sistemi informativi aziendali in corso.",
    "Report risorse umane: Stato delle assunzioni e piano di formazione continua."
]


@dataclass
class TestCase:
    test_id: str
    category: str
    variant: int
    doc_length: int
    question: str
    expected_answer: str
    document_segments: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_needle_tests(n_variants: int = 10, lengths: List[int] = None) -> List[TestCase]:
    """Generate needle-in-a-haystack test cases."""
    if lengths is None:
        lengths = [20, 50, 100, 200]
    
    tests = []
    for variant in range(n_variants):
        rng = random.Random(42 + variant)
        
        # Pick unique secrets for this variant
        secret_type, secret_template = SECRET_TYPES[variant % len(SECRET_TYPES)]
        name = PERSON_NAMES[rng.randint(0, len(PERSON_NAMES) - 1)]
        code = f"{rng.randint(1000,9999)}-{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}-{rng.randint(1000,9999)}"
        date = f"{rng.randint(1,28)} {'gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre'.split()[rng.randint(0,11)]}"
        amount = f"{rng.randint(1,50) * 100000}"
        
        needle_text = secret_template.format(
            name=name, code=code, date=date, time=f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}",
            place=f"molo {rng.randint(1,20)}", amount=amount, floor=str(rng.randint(1,10))
        )
        
        # Extract key answer token for matching
        answer_key = needle_text
        
        # Question templates
        question_map = {
            "informatore": f"Chi è l'informatore menzionato nel documento (variante {variant})?",
            "password": f"Qual è la password master del sistema (variante {variant})?",
            "codice": f"Qual è il codice di accesso al vault (variante {variant})?",
            "appuntamento": f"Quando e dove è fissato l'appuntamento segreto (variante {variant})?",
            "trasferimento": f"Qual è l'importo del trasferimento fondi autorizzato (variante {variant})?",
            "chiave_api": f"Qual è la chiave API nel documento (variante {variant})?",
            "frequenza": f"Qual è la frequenza radio segreta (variante {variant})?",
            "protocollo": f"Qual è il protocollo di emergenza (variante {variant})?",
            "contatto": f"Chi è il contatto di riferimento (variante {variant})?",
            "indirizzo": f"Qual è l'indirizzo sicuro (variante {variant})?"
        }
        question = question_map.get(secret_type, f"Trova l'informazione segreta (variante {variant})")
        
        for length in lengths:
            # Place needle at a random position
            needle_pos = rng.randint(0, length - 1)
            segments = []
            for i in range(length):
                if i == needle_pos:
                    segments.append(f"[SEGMENTO {i}] {needle_text}")
                else:
                    segments.append(f"[SEGMENTO {i}] {FILLER_TEXTS[i % len(FILLER_TEXTS)]}")
            
            tests.append(TestCase(
                test_id=f"needle_v{variant}_L{length}",
                category="needle",
                variant=variant,
                doc_length=length,
                question=question,
                expected_answer=answer_key,
                document_segments=segments,
                metadata={"needle_pos": needle_pos, "secret_type": secret_type}
            ))
    
    return tests


def generate_multi_hop_tests(n_variants: int = 10, lengths: List[int] = None) -> List[TestCase]:
    """Generate multi-hop reasoning test cases."""
    if lengths is None:
        lengths = [20, 50, 100, 200]
    
    tests = []
    for variant in range(n_variants):
        rng = random.Random(100 + variant)
        
        # Pick entities
        p1_idx = variant % len(PERSON_NAMES)
        p2_idx = (variant + 5) % len(PERSON_NAMES)
        c1_idx = variant % len(COMPANY_NAMES)
        c2_idx = (variant + 3) % len(COMPANY_NAMES)
        city1 = CITY_DATA[variant % len(CITY_DATA)]
        city2 = CITY_DATA[(variant + 4) % len(CITY_DATA)]
        sector1 = SECTORS[variant % len(SECTORS)]
        sector2 = SECTORS[(variant + 3) % len(SECTORS)]
        year1 = 2005 + rng.randint(0, 15)
        year2 = 2005 + rng.randint(0, 15)
        
        person1 = PERSON_NAMES[p1_idx]
        person2 = PERSON_NAMES[p2_idx]
        company1 = COMPANY_NAMES[c1_idx]
        company2 = COMPANY_NAMES[c2_idx]
        
        fact_segments = [
            f"Profile: {person1} ricopre il ruolo di Direttore Tecnico presso {company1}.",
            f"Azienda: {company1} opera nel settore {sector1} con sede principale a {city1[0]}. Fondata nel {year1}.",
            f"Citta: {city1[0]}, capoluogo della regione {city1[1]}. Popolazione: {city1[2]} abitanti.",
            f"Profile: {person2} ricopre il ruolo di Responsabile Marketing presso {company2}.",
            f"Azienda: {company2} opera nel settore {sector2} con sede principale a {city2[0]}. Fondata nel {year2}.",
            f"Citta: {city2[0]}, capoluogo della regione {city2[1]}. Popolazione: {city2[2]} abitanti.",
        ]
        
        # Questions and answers  
        qa_pairs = [
            (f"Dove lavora {person1} e in quale città?", f"{company1} a {city1[0]}"),
            (f"Quanti abitanti ha la città dove lavora {person1}?", str(city1[2])),
            (f"Quando è stata fondata {company2}?", str(year2)),
        ]
        
        for q_idx, (question, expected) in enumerate(qa_pairs):
            for length in lengths:
                segments = []
                fact_positions = [0, 1, 2, max(5, length // 4), max(6, length // 4 + 1), max(7, length // 4 + 2)]
                
                for i in range(length):
                    placed = False
                    for f_idx, f_pos in enumerate(fact_positions):
                        if i == f_pos and f_idx < len(fact_segments):
                            segments.append(f"[SEGMENTO {i}] {fact_segments[f_idx]}")
                            placed = True
                            break
                    if not placed:
                        segments.append(f"[SEGMENTO {i}] {FILLER_TEXTS[i % len(FILLER_TEXTS)]}")
                
                tests.append(TestCase(
                    test_id=f"multi_hop_v{variant}_q{q_idx}_L{length}",
                    category="multi_hop",
                    variant=variant,
                    doc_length=length,
                    question=question,
                    expected_answer=expected,
                    document_segments=segments
                ))
    
    return tests


def generate_temporal_tests(n_variants: int = 10, lengths: List[int] = None) -> List[TestCase]:
    """Generate temporal reasoning test cases."""
    if lengths is None:
        lengths = [20, 50, 100, 200]
    
    tests = []
    for variant in range(n_variants):
        rng = random.Random(200 + variant)
        
        base_year = 2015 + rng.randint(0, 8)
        company_name = COMPANY_NAMES[variant % len(COMPANY_NAMES)]
        
        events = [
            {"data": f"{base_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}", 
             "evento": f"Fondazione {company_name}", "tipo": "business"},
            {"data": f"{base_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
             "evento": f"{company_name} riceve seed funding", "tipo": "finance"},
            {"data": f"{base_year+1}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
             "evento": f"Lancio prodotto di {company_name}", "tipo": "product"},
            {"data": f"{base_year+2}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
             "evento": f"{company_name} acquisisce partner strategico", "tipo": "business"},
            {"data": f"{base_year+3}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
             "evento": f"{company_name} supera 1 milione utenti", "tipo": "milestone"},
        ]
        
        qa_pairs = [
            (f"Quando è stata fondata {company_name}?", events[0]["data"]),
            (f"Quando {company_name} ha ricevuto il seed funding?", events[1]["data"]),
            (f"Quando {company_name} ha acquisito il partner strategico?", events[3]["data"]),
        ]
        
        for q_idx, (question, expected) in enumerate(qa_pairs):
            for length in lengths:
                segments = []
                for i in range(length):
                    if i < len(events):
                        ev = events[i]
                        segments.append(f"[SEGMENTO {i}] Data: {ev['data']}. Evento: {ev['evento']}. Tipo: {ev['tipo']}.")
                    else:
                        segments.append(f"[SEGMENTO {i}] Update periodico sulle operazioni aziendali in corso.")
                
                tests.append(TestCase(
                    test_id=f"temporal_v{variant}_q{q_idx}_L{length}",
                    category="temporal",
                    variant=variant,
                    doc_length=length,
                    question=question,
                    expected_answer=expected,
                    document_segments=segments
                ))
    
    return tests


def generate_aggregation_tests(n_variants: int = 10, lengths: List[int] = None) -> List[TestCase]:
    """Generate aggregation test cases with varied financial data."""
    if lengths is None:
        lengths = [20, 50, 100, 200]
    
    tests = []
    for variant in range(n_variants):
        rng = random.Random(300 + variant)
        
        n_transactions = rng.randint(4, 8)
        transactions = []
        descriptions_income = ["Vendita prodotto", "Contratto cliente", "Servizi consulenza", 
                                "Licenze software", "Ricavi pubblicitari", "Commissioni"]
        descriptions_expense = ["Acquisto materie prime", "Spese operative", "Stipendi personale",
                                 "Utility e affitti", "Marketing", "Formazione"]
        
        for t_id in range(n_transactions):
            is_income = t_id % 2 == 0
            amount = rng.randint(3, 50) * 1000
            desc = rng.choice(descriptions_income if is_income else descriptions_expense)
            transactions.append({
                "id": t_id + 1,
                "importo": amount,
                "tipo": "entrata" if is_income else "uscita",
                "descrizione": f"{desc} {chr(65 + t_id)}"
            })
        
        total_income = sum(t["importo"] for t in transactions if t["tipo"] == "entrata")
        total_expense = sum(t["importo"] for t in transactions if t["tipo"] == "uscita")
        net = total_income - total_expense
        
        qa_pairs = [
            (f"Qual è il totale delle entrate?", str(total_income)),
            (f"Qual è il totale delle uscite?", str(total_expense)),
            (f"Qual è il bilancio netto?", str(net)),
        ]
        
        for q_idx, (question, expected) in enumerate(qa_pairs):
            for length in lengths:
                segments = []
                for i in range(length):
                    if i < len(transactions):
                        t = transactions[i]
                        tipo_en = "income" if t["tipo"] == "entrata" else "expense"
                        segments.append(f"[SEGMENT {i}] Transaction #{t['id']}: {t['descrizione']}. Amount: €{t['importo']:.2f} ({tipo_en}).")
                    else:
                        segments.append(f"[SEGMENT {i}] Accounting record: routine operation registered in the system.")
                
                tests.append(TestCase(
                    test_id=f"aggregation_v{variant}_q{q_idx}_L{length}",
                    category="aggregation",
                    variant=variant,
                    doc_length=length,
                    question=question,
                    expected_answer=expected,
                    document_segments=segments
                ))
    
    return tests


def generate_entity_tests(n_variants: int = 10, lengths: List[int] = None) -> List[TestCase]:
    """Generate entity tracking test cases."""
    if lengths is None:
        lengths = [20, 50, 100, 200]
    
    tests = []
    for variant in range(n_variants):
        rng = random.Random(400 + variant)
        
        project_names = [f"Progetto {chr(65 + i)}" for i in range(3)]
        phases = ["concept", "pianificazione", "sviluppo", "testing", "deployment", "lancio"]
        
        projects = []
        for p_name in project_names:
            start_phase = rng.choice(phases[:3])
            end_phase = rng.choice(phases[3:])
            duration = rng.randint(6, 36)
            projects.append({
                "nome": p_name,
                "fase_iniziale": start_phase,
                "fase_finale": end_phase,
                "durata_mesi": duration
            })
        
        longest = max(projects, key=lambda p: p["durata_mesi"])
        avg_duration = sum(p["durata_mesi"] for p in projects) / len(projects)
        launched = sum(1 for p in projects if p["fase_finale"] == "lancio")
        
        qa_pairs = [
            (f"Qual è il progetto più lungo?", longest["nome"]),
            (f"Qual è la durata media dei progetti?", str(avg_duration)),
            (f"Quanti progetti hanno raggiunto la fase di lancio?", str(launched)),
        ]
        
        for q_idx, (question, expected) in enumerate(qa_pairs):
            for length in lengths:
                segments = []
                for i in range(length):
                    if i < len(projects):
                        p = projects[i]
                        segments.append(f"[SEGMENTO {i}] {p['nome']}: Fase iniziale={p['fase_iniziale']}, Fase finale={p['fase_finale']}, Durata={p['durata_mesi']} mesi.")
                    else:
                        segments.append(f"[SEGMENTO {i}] Report avanzamento progetti: aggiornamenti periodici sullo stato di avanzamento.")
                
                tests.append(TestCase(
                    test_id=f"entity_v{variant}_q{q_idx}_L{length}",
                    category="entity",
                    variant=variant,
                    doc_length=length,
                    question=question,
                    expected_answer=expected,
                    document_segments=segments
                ))
    
    return tests


# ═══════════════════════════════════════════════════════════════════
#  ANSWER EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_answer(actual: str, expected: str, category: str) -> bool:
    """Evaluate if the answer is correct."""
    actual_lower = actual.lower()
    expected_lower = expected.lower()
    
    if category == "needle":
        keywords = [w for w in expected_lower.split() if len(w) > 4][:3]
        return any(k in actual_lower for k in keywords)
    
    elif category == "multi_hop":
        return expected_lower in actual_lower or any(
            p in actual_lower for p in expected_lower.split() if len(p) > 3
        )
    
    elif category == "temporal":
        return expected_lower in actual_lower
    
    elif category == "aggregation":
        try:
            exp_num = int(float(expected))
            return str(exp_num) in actual or f"{float(expected):.2f}" in actual
        except ValueError:
            return expected_lower in actual_lower
    
    elif category == "entity":
        try:
            exp_num = float(expected)
            return str(int(exp_num)) in actual or str(exp_num) in actual
        except ValueError:
            return expected_lower in actual_lower
    
    return False


# ═══════════════════════════════════════════════════════════════════
#  BASELINES
# ═══════════════════════════════════════════════════════════════════

def run_bm25_baseline(test: TestCase) -> Tuple[str, float]:
    """Baseline 1: BM25 retrieval only — no agents, no arena."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return "[BM25] rank_bm25 not installed", 0.0
    
    tokenized_corpus = [seg.lower().split() for seg in test.document_segments]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = test.question.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    top_segments = [test.document_segments[i] for i in top_idx]
    
    return "\n".join(top_segments), max(scores) if len(scores) > 0 else 0.0


def run_single_agent_baseline(
    test: TestCase,
    embedding_generator=None,
    device=None
) -> Tuple[str, float, float]:
    """Baseline 2: Single-Agent BAS — no Arena combat, no Ziora."""
    studsar_manager = None
    if embedding_generator is not None:
        studsar_manager = StudSarManager(
            embedding_generator=embedding_generator,
            device=device,
            quiet=True
        )
    bas = BASEngine(
        max_agents=200,
        agents_per_query=1,
        countdown_seconds=30.0,
        judge_confidence_threshold=0.6,
        auto_god=True,
        dynamic_agent_selection=True,
        enable_red_agent=False,
        quiet=True,
        studsar_manager=studsar_manager
    )
    
    document = "\n\n".join(test.document_segments)
    bas.ingest_document(document, f"single_{test.test_id}", segment_length=100)
    
    result = bas.query(test.question, max_rounds=1)
    return result.final_answer, result.judge_confidence, result.memory_coherence_score


# ═══════════════════════════════════════════════════════════════════
#  MAIN BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    test_id: str
    category: str
    doc_length: int
    question: str
    expected_answer: str
    system: str  # "bas", "bm25", "single_agent"
    actual_answer: str
    is_correct: bool
    confidence: float
    latency_seconds: float
    memory_coherence: float
    timestamp: str
    
    def to_dict(self):
        return {
            "test_id": self.test_id,
            "category": self.category,
            "doc_length": self.doc_length,
            "question": self.question[:100],
            "expected_answer": self.expected_answer[:100],
            "system": self.system,
            "actual_answer": self.actual_answer[:200],
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "latency_seconds": self.latency_seconds,
            "memory_coherence": self.memory_coherence,
            "timestamp": self.timestamp
        }


class LOCOMOBenchmarkV2:
    """Scaled LOCOMO Benchmark for BAS with baselines."""
    
    def __init__(self, n_variants: int = 10, lengths: List[int] = None,
                 run_baselines: bool = True):
        self.n_variants = n_variants
        self.lengths = lengths or [20, 50, 100, 200]
        self.run_baselines = run_baselines
        self.results: List[BenchmarkResult] = []
        self._shared_embedding_generator = None
        self._shared_device = None

    def _ensure_shared_embedding(self) -> None:
        if self._shared_embedding_generator is not None:
            return
        manager = StudSarManager(quiet=True)
        self._shared_embedding_generator = manager.embedding_generator
        self._shared_device = manager.device
        
    def generate_all_tests(self) -> List[TestCase]:
        """Generate all test cases."""
        tests = []
        tests.extend(generate_needle_tests(self.n_variants, self.lengths))
        tests.extend(generate_multi_hop_tests(self.n_variants, self.lengths))
        tests.extend(generate_temporal_tests(self.n_variants, self.lengths))
        tests.extend(generate_aggregation_tests(self.n_variants, self.lengths))
        tests.extend(generate_entity_tests(self.n_variants, self.lengths))
        return tests
    
    def run_bas_test(self, test: TestCase) -> BenchmarkResult:
        """Run a single test with full BAS system."""
        self._ensure_shared_embedding()
        studsar_manager = StudSarManager(
            embedding_generator=self._shared_embedding_generator,
            device=self._shared_device,
            quiet=True
        )
        bas = BASEngine(
            max_agents=200,
            agents_per_query=10,
            countdown_seconds=30.0,
            judge_confidence_threshold=0.6,
            auto_god=True,
            dynamic_agent_selection=True,
            quiet=True,
            studsar_manager=studsar_manager
        )
        
        document = "\n\n".join(test.document_segments)
        bas.ingest_document(document, f"bas_{test.test_id}", segment_length=100)
        
        start = time.time()
        result = bas.query(test.question, max_rounds=3)
        latency = time.time() - start
        
        is_correct = evaluate_answer(result.final_answer, test.expected_answer, test.category)
        
        return BenchmarkResult(
            test_id=test.test_id,
            category=test.category,
            doc_length=test.doc_length,
            question=test.question,
            expected_answer=test.expected_answer,
            system="bas",
            actual_answer=result.final_answer[:200],
            is_correct=is_correct,
            confidence=result.judge_confidence,
            latency_seconds=latency,
            memory_coherence=result.memory_coherence_score,
            timestamp=datetime.now().isoformat()
        )
    
    def run_bm25_test(self, test: TestCase) -> BenchmarkResult:
        """Run a single test with BM25 baseline."""
        start = time.time()
        answer, confidence = run_bm25_baseline(test)
        latency = time.time() - start
        
        is_correct = evaluate_answer(answer, test.expected_answer, test.category)
        
        return BenchmarkResult(
            test_id=test.test_id,
            category=test.category,
            doc_length=test.doc_length,
            question=test.question,
            expected_answer=test.expected_answer,
            system="bm25",
            actual_answer=answer[:200],
            is_correct=is_correct,
            confidence=confidence,
            latency_seconds=latency,
            memory_coherence=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def run_single_agent_test(self, test: TestCase) -> BenchmarkResult:
        """Run a single test with single-agent baseline."""
        self._ensure_shared_embedding()
        start = time.time()
        answer, confidence, coherence = run_single_agent_baseline(
            test,
            embedding_generator=self._shared_embedding_generator,
            device=self._shared_device
        )
        latency = time.time() - start
        
        is_correct = evaluate_answer(answer, test.expected_answer, test.category)
        
        return BenchmarkResult(
            test_id=test.test_id,
            category=test.category,
            doc_length=test.doc_length,
            question=test.question,
            expected_answer=test.expected_answer,
            system="single_agent",
            actual_answer=answer[:200],
            is_correct=is_correct,
            confidence=confidence,
            latency_seconds=latency,
            memory_coherence=coherence,
            timestamp=datetime.now().isoformat()
        )
    
    def run_full_benchmark(self, checkpoint_every: int = 50):
        """Run the complete benchmark suite."""
        tests = self.generate_all_tests()
        total = len(tests)
        systems = ["bas"]
        if self.run_baselines:
            systems.extend(["bm25", "single_agent"])
        
        print("=" * 70)
        print(f"  LOCOMO BENCHMARK V2 — SCALED")
        print(f"  Total test cases: {total}")
        print(f"  Systems: {', '.join(systems)}")
        print(f"  Categories: needle, multi_hop, temporal, aggregation, entity")
        print(f"  Variants: {self.n_variants}, Lengths: {self.lengths}")
        print("=" * 70)
        
        checkpoint_file = os.path.join(os.path.dirname(__file__), "locomo_v2_checkpoint.json")
        
        for sys_name in systems:
            print(f"\n{'=' * 50}")
            print(f"  Running system: {sys_name.upper()}")
            print(f"{'=' * 50}")
            
            for idx, test in enumerate(tests):
                # Progress
                pct = (idx + 1) / total * 100
                print(f"  [{sys_name}] {idx+1}/{total} ({pct:.1f}%) — {test.test_id}", end="", flush=True)
                
                try:
                    if sys_name == "bas":
                        result = self.run_bas_test(test)
                    elif sys_name == "bm25":
                        result = self.run_bm25_test(test)
                    elif sys_name == "single_agent":
                        result = self.run_single_agent_test(test)
                    
                    self.results.append(result)
                    status = "OK" if result.is_correct else "FAIL"
                    print(f" — {status} ({result.latency_seconds:.2f}s)")
                except Exception as e:
                    print(f" — ERROR: {e}")
                    self.results.append(BenchmarkResult(
                        test_id=test.test_id, category=test.category,
                        doc_length=test.doc_length, question=test.question,
                        expected_answer=test.expected_answer, system=sys_name,
                        actual_answer=f"ERROR: {str(e)[:100]}", is_correct=False,
                        confidence=0.0, latency_seconds=0.0, memory_coherence=0.0,
                        timestamp=datetime.now().isoformat()
                    ))
                
                # Checkpoint
                if (idx + 1) % checkpoint_every == 0:
                    self._save_checkpoint(checkpoint_file)
        
        return self.results
    
    def _save_checkpoint(self, filepath: str):
        """Save intermediate results."""
        data = {"results": [r.to_dict() for r in self.results], "timestamp": datetime.now().isoformat()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Print comparison summary across all systems."""
        print("\n" + "=" * 80)
        print("  LOCOMO BENCHMARK V2 — RESULTS COMPARISON")
        print("=" * 80)
        
        # Group by system
        systems = {}
        for r in self.results:
            if r.system not in systems:
                systems[r.system] = []
            systems[r.system].append(r)
        
        # Overall accuracy per system
        print(f"\n{'System':<20} {'Total':<8} {'Correct':<10} {'Accuracy':<12} {'Avg Latency':<12}")
        print("-" * 62)
        for sys_name, results in sorted(systems.items()):
            total = len(results)
            correct = sum(1 for r in results if r.is_correct)
            acc = correct / total * 100 if total > 0 else 0
            avg_lat = sum(r.latency_seconds for r in results) / total if total > 0 else 0
            print(f"  {sys_name:<18} {total:<8} {correct:<10} {acc:<12.1f}% {avg_lat:<12.2f}s")
        
        # Per-category breakdown for each system
        print(f"\n{'Category':<15}", end="")
        for sys_name in sorted(systems.keys()):
            print(f" {sys_name:<15}", end="")
        print()
        print("-" * (15 + 15 * len(systems)))
        
        categories = sorted(set(r.category for r in self.results))
        for cat in categories:
            print(f"  {cat:<13}", end="")
            for sys_name in sorted(systems.keys()):
                cat_results = [r for r in systems[sys_name] if r.category == cat]
                if cat_results:
                    correct = sum(1 for r in cat_results if r.is_correct)
                    acc = correct / len(cat_results) * 100
                    print(f" {acc:>5.1f}%({correct}/{len(cat_results)})", end="")
                else:
                    print(f" {'N/A':>13}", end="")
            print()
        
        # Confidence-error anticorrelation analysis (BAS only)
        bas_results = systems.get("bas", [])
        if bas_results:
            correct_confs = [r.confidence for r in bas_results if r.is_correct]
            incorrect_confs = [r.confidence for r in bas_results if not r.is_correct]
            
            if correct_confs and incorrect_confs:
                avg_correct = sum(correct_confs) / len(correct_confs)
                avg_incorrect = sum(incorrect_confs) / len(incorrect_confs)
                ratio = avg_correct / avg_incorrect if avg_incorrect > 0 else float('inf')
                
                print(f"\n  Confidence-Error Anticorrelation (BAS):")
                print(f"    Mean confidence (correct):   {avg_correct:.6f}")
                print(f"    Mean confidence (incorrect): {avg_incorrect:.6f}")
                print(f"    Ratio: {ratio:.2f}x")
        
        # Save results
        output_file = os.path.join(os.path.dirname(__file__), "locomo_v2_results.json")
        summary = {
            "summary": {
                sys_name: {
                    "total": len(results),
                    "correct": sum(1 for r in results if r.is_correct),
                    "accuracy": sum(1 for r in results if r.is_correct) / len(results) * 100 if results else 0,
                    "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
                    "avg_latency": sum(r.latency_seconds for r in results) / len(results) if results else 0,
                }
                for sys_name, results in systems.items()
            },
            "by_category": {
                cat: {
                    sys_name: {
                        "correct": sum(1 for r in systems.get(sys_name, []) if r.category == cat and r.is_correct),
                        "total": sum(1 for r in systems.get(sys_name, []) if r.category == cat),
                        "accuracy": (sum(1 for r in systems.get(sys_name, []) if r.category == cat and r.is_correct) /
                                     max(1, sum(1 for r in systems.get(sys_name, []) if r.category == cat))) * 100
                    }
                    for sys_name in systems
                }
                for cat in categories
            },
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n  Results saved to: {output_file}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LOCOMO Benchmark V2 for BAS")
    parser.add_argument("--variants", type=int, default=10, help="Number of variants per category")
    parser.add_argument("--lengths", type=int, nargs="+", default=[20, 50, 100], help="Document lengths to test")
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline comparisons")
    parser.add_argument("--bas-only", action="store_true", help="Run only full BAS system")
    args = parser.parse_args()
    
    benchmark = LOCOMOBenchmarkV2(
        n_variants=args.variants,
        lengths=args.lengths,
        run_baselines=not args.no_baselines and not args.bas_only
    )
    
    benchmark.run_full_benchmark()
    benchmark.print_summary()
