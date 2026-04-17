--- locomo_benchmark.py (原始)


+++ locomo_benchmark.py (修改后)
"""
LOCOMO Benchmark per BAS (Brain Agent Supreme)

Basato su: Long Context Memory benchmark — Meta AI Research, 2024
https://github.com/facebookresearch/locomo

Questo benchmark testa la capacità di BAS di:
1. Needle in a Haystack - Trovare informazioni specifiche in documenti lunghi
2. Multi-hop Reasoning - Collegare informazioni da segmenti diversi
3. Temporal Reasoning - Capire sequenze e timeline
4. Aggregation - Sommare/calcolare dati distribuiti
5. Entity Tracking - Tracciare entità attraverso il documento

METRICHE CHIAVE:
- Recall@K: Capacità di recuperare informazioni rilevanti
- Precision: Accuratezza delle risposte
- F1-Score: Bilanciamento recall/precision
- Latency: Tempo di risposta
- Memory Coherence: Coerenza della memoria StudSar
"""

import sys
import os
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Aggiungi src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.arena import BASEngine, SegmentAgent
from src.studsar import StudSarManager


@dataclass
class BenchmarkResult:
    """Risultato di un singolo test del benchmark."""
    test_name: str
    test_type: str  # needle, multi_hop, temporal, aggregation, entity
    document_length: int  # numero di segmenti
    question: str
    expected_answer: str
    actual_answer: str
    is_correct: bool
    confidence: float
    latency_seconds: float
    agents_used: int
    memory_coherence: float
    timestamp: str


@dataclass
class BenchmarkSummary:
    """Riepilogo completo del benchmark."""
    total_tests: int
    passed: int
    failed: int
    accuracy: float
    avg_latency: float
    avg_agents_used: float
    avg_memory_coherence: float
    results_by_type: Dict[str, Dict[str, float]]
    detailed_results: List[BenchmarkResult]
    timestamp: str


class LOCOMOBenchmark:
    """
    Implementazione del benchmark LOCOMO per BAS.

    I test sono progettati per valutare:
    1. Recupero informazioni specifiche (Needle in a Haystack)
    2. Ragionamento multi-hop (collegare fatti distanti)
    3. Ragionamento temporale (sequenze, prima/dopo)
    4. Aggregazione (somme, conteggi, medie)
    5. Tracciamento entità (persone, luoghi, oggetti)
    """

    def __init__(self, bas_engine: BASEngine, studsar: StudSarManager):
        self.bas = bas_engine
        self.studsar = studsar
        self.results: List[BenchmarkResult] = []

    def generate_test_document(self, test_type: str, length: int = 50) -> Tuple[str, Dict]:
        """
        Genera un documento di test strutturato per un tipo specifico di test.

        Returns:
            Tuple[documento_testo, metadati_con_risposte_corrette]
        """

        if test_type == "needle":
            return self._generate_needle_document(length)
        elif test_type == "multi_hop":
            return self._generate_multi_hop_document(length)
        elif test_type == "temporal":
            return self._generate_temporal_document(length)
        elif test_type == "aggregation":
            return self._generate_aggregation_document(length)
        elif test_type == "entity":
            return self._generate_entity_document(length)
        else:
            raise ValueError(f"Tipo di test sconosciuto: {test_type}")

    def _generate_needle_document(self, length: int) -> Tuple[str, Dict]:
        """
        Genera documento per test 'Needle in a Haystack'.
        Informazioni critiche nascoste in posizioni casuali.
        """
        segments = []
        metadata = {"answers": {}, "needle_positions": {}}

        # Nomi e valori speciali da nascondere
        special_facts = {
            "codice_segreto": "X7K9M2P4",
            "nome_informatore": "Dott.ssa Elena Marchetti",
            "luogo_incontro": "Biblioteca Nazionale di Firenze, sala manoscritti",
            "password_accesso": "Prometeo1984!",
            "numero_conto": "CH93 0076 2011 6238 5295 7",
            "data_lancio": "15 marzo 2025, ore 14:30 UTC",
            "chiave_crittografia": "AES-256-GCM-SHA384-RSA4096",
            "contatto_emergenza": "+39 333 7829461"
        }

        # Posizioni casuali per le informazioni critiche
        import random
        available_positions = list(range(length))
        random.seed(42)  # Riproducibilità

        for fact_name, fact_value in special_facts.items():
            if available_positions:
                pos = random.choice(available_positions)
                available_positions.remove(pos)
                metadata["needle_positions"][fact_name] = pos

        # Genera segmenti
        for i in range(length):
            if i in metadata["needle_positions"].values():
                # Trova quale fatto va in questa posizione
                for fact_name, pos in metadata["needle_positions"].items():
                    if pos == i:
                        fact_value = special_facts[fact_name]
                        segments.append(f"[SEGMENTO {i}] Informazione critica: Il {fact_name.replace('_', ' ')} è '{fact_value}'. Questo dato è cruciale per l'operazione.")
                        metadata["answers"][fact_name] = fact_value
                        break
            else:
                # Testo di riempimento coerente ma irrilevante
                filler_texts = [
                    f"[SEGMENTO {i}] L'analisi dei dati mostra tendenze significative nel comportamento degli utenti durante il periodo osservato.",
                    f"[SEGMENTO {i}] I protocolli di sicurezza standard sono stati implementati secondo le linee guida internazionali ISO 27001.",
                    f"[SEGMENTO {i}] La documentazione tecnica include specifiche dettagliate sui requisiti funzionali e non funzionali del sistema.",
                    f"[SEGMENTO {i}] Le metriche di performance indicano un miglioramento del 23% rispetto al trimestre precedente.",
                    f"[SEGMENTO {i}] Il team di sviluppo ha completato la fase di testing intensivo su tutti i moduli principali.",
                ]
                segments.append(filler_texts[i % len(filler_texts)])

        document = "\n\n".join(segments)
        metadata["document_length"] = length
        metadata["num_needles"] = len(special_facts)

        return document, metadata

    def _generate_multi_hop_document(self, length: int) -> Tuple[str, Dict]:
        """
        Genera documento per test 'Multi-hop Reasoning'.
        Richiede collegare informazioni da segmenti diversi per dedurre risposte.
        """
        segments = []
        metadata = {"facts": {}, "reasoning_chains": {}, "answers": {}}

        # Crea una storia coerente con fatti distribuiti
        # Esempio: Persona X lavora in azienda Y, azienda Y ha sede in città Z,
        # città Z ha popolazione W → Domanda: Dove lavora X e quanti abitanti ha quella città?

        entities = {
            "persona_1": {"nome": "Marco Rossi", "ruolo": "Direttore Tecnico", "azienda": "TechCorp Italia"},
            "persona_2": {"nome": "Laura Bianchi", "ruolo": "Responsabile Marketing", "azienda": "Innovazione SpA"},
            "persona_3": {"nome": "Giuseppe Verdi", "ruolo": "CEO", "azienda": "DataSolutions Srl"},
            "azienda_1": {"nome": "TechCorp Italia", "settore": "Software", "citta": "Milano", "fondazione": 2010},
            "azienda_2": {"nome": "Innovazione SpA", "settore": "Consulenza", "citta": "Roma", "fondazione": 2015},
            "azienda_3": {"nome": "DataSolutions Srl", "settore": "Analytics", "citta": "Torino", "fondazione": 2018},
            "citta_1": {"nome": "Milano", "regione": "Lombardia", "abitanti": 1396000, "sindaco": "Sala"},
            "citta_2": {"nome": "Roma", "regione": "Lazio", "abitanti": 2873000, "sindaco": "Gualtieri"},
            "citta_3": {"nome": "Torino", "regione": "Piemonte", "abitanti": 875000, "sindaco": "Lo Russo"},
        }

        metadata["facts"] = entities

        # Catene di ragionamento
        metadata["reasoning_chains"] = {
            "dove_lavora_marco": ["persona_1→azienda", "azienda_1→città"],
            "abitanti_citta_marco": ["persona_1→azienda", "azienda_1→città", "città_1→abitanti"],
            "chi_ceo_datasolutions": ["azienda_3→CEO"],
            "anno_fondazione_innovazione": ["azienda_2→fondazione"],
            "sindaco_torino": ["città_3→sindaco"],
        }

        # Risposte corrette
        metadata["answers"] = {
            "dove_lavora_marco": "TechCorp Italia a Milano",
            "abitanti_citta_marco": 1396000,
            "chi_ceo_datasolutions": "Giuseppe Verdi",
            "anno_fondazione_innovazione": 2015,
            "sindaco_torino": "Lo Russo",
        }

        # Genera segmenti distribuendo le informazioni
        segment_templates = [
            "[SEGMENTO {i}] Profile: {persona[nome]} ricopre il ruolo di {persona[ruolo]} presso {persona[azienda]}.",
            "[SEGMENTO {i}] Azienda: {azienda[nome]} opera nel settore {azienda[settore]} con sede principale a {azienda[citta]}. Fondata nel {azienda[fondazione]}.",
            "[SEGMENTO {i}] Citta: {citta[nome]}, capoluogo della regione {citta[regione]}. Popolazione: {citta[abitanti]} abitanti. Sindaco: {citta[sindaco]}.",
            "[SEGMENTO {i}] Analisi di mercato: Il settore tecnologico mostra crescita costante nel Nord Italia.",
            "[SEGMENTO {i}] Report finanziario: I dati consolidati mostrano performance positive across all sectors.",
            "[SEGMENTO {i}] Nota operativa: Procedure standardizzate per tutti i dipartimenti.",
        ]

        facts_to_distribute = [
            ("persona", "persona_1", 0),
            ("azienda", "azienda_1", 1),
            ("citta", "citta_1", 2),
            ("persona", "persona_2", 5),
            ("azienda", "azienda_2", 6),
            ("citta", "citta_2", 7),
            ("persona", "persona_3", 10),
            ("azienda", "azienda_3", 11),
            ("citta", "citta_3", 12),
        ]

        for i in range(length):
            placed = False
            for entity_type, entity_key, target_pos in facts_to_distribute:
                if i == target_pos and target_pos < length:
                    # Estrai il numero dall'entity_key (es: "persona_1" → "1")
                    entity_num = entity_key.split("_")[-1]

                    # Seleziona il template corretto
                    if entity_type == "persona":
                        entity_data = entities[f"persona_{entity_num}"]
                        template = segment_templates[0]
                        segments.append(template.format(i=i, persona=entity_data))
                    elif entity_type == "azienda":
                        entity_data = entities[f"azienda_{entity_num}"]
                        template = segment_templates[1]
                        segments.append(template.format(i=i, azienda=entity_data))
                    elif entity_type == "citta":
                        entity_data = entities[f"citta_{entity_num}"]
                        template = segment_templates[2]
                        segments.append(template.format(i=i, citta=entity_data))
                    placed = True
                    break

            if not placed:
                segments.append(segment_templates[i % len(segment_templates)].format(i=i, persona={}, azienda={}, citta={}))

        document = "\n\n".join(segments)
        metadata["document_length"] = length

        return document, metadata

    def _generate_temporal_document(self, length: int) -> Tuple[str, Dict]:
        """
        Genera documento per test 'Temporal Reasoning'.
        Eventi distribuiti nel tempo con relazioni temporali complesse.
        """
        segments = []
        metadata = {"events": [], "timeline": {}, "answers": {}}

        # Timeline di eventi
        events = [
            {"data": "2020-01-15", "evento": "Fondazione startup Alpha", "tipo": "business"},
            {"data": "2020-06-20", "evento": "Alpha riceve seed funding da Beta Ventures", "tipo": "finance"},
            {"data": "2021-03-10", "evento": "Lancio prodotto flagship di Alpha", "tipo": "product"},
            {"data": "2021-09-05", "evento": "Alpha acquisisce Gamma Tech", "tipo": "business"},
            {"data": "2022-02-28", "evento": "Gamma Tech diventa divisione di Alpha", "tipo": "business"},
            {"data": "2022-11-15", "evento": "Alpha supera 1 milione di utenti", "tipo": "milestone"},
            {"data": "2023-04-22", "evento": "Alpha lancia versione 2.0 del prodotto", "tipo": "product"},
            {"data": "2023-08-30", "evento": "Beta Ventures vende partecipazione in Alpha", "tipo": "finance"},
            {"data": "2024-01-10", "evento": "Alpha annuncia espansione internazionale", "tipo": "business"},
            {"data": "2024-06-15", "evento": "Alpha apre uffici a Londra e Berlino", "tipo": "business"},
        ]

        metadata["events"] = events
        metadata["timeline"] = {e["data"]: e["evento"] for e in events}

        # Domande e risposte
        metadata["answers"] = {
            "primo_evento": "Fondazione startup Alpha (2020-01-15)",
            "ultimo_evento": "Alpha apre uffici a Londra e Berlino (2024-06-15)",
            "anni_durata": 4,  # dal 2020 al 2024
            "acquisizione_gamma": "2021-09-05",
            "eventi_2022": 2,  # due eventi nel 2022
            "prima_del_2022": 4,  # eventi prima del 2022
            "dopo_acquisizione": "Gamma Tech diventa divisione di Alpha (2022-02-28)",
        }

        # Genera segmenti
        for i in range(length):
            if i < len(events):
                event = events[i]
                segments.append(
                    f"[SEGMENTO {i}] TIMELINE EVENT: Data {event['data']}. "
                    f"{event['evento']}. Categoria: {event['tipo']}. "
                    f"Questo evento segna un punto importante nella storia aziendale."
                )
            else:
                filler = f"[SEGMENTO {i}] Contesto aggiuntivo: Informazioni di supporto sull'ecosistema startup italiano ed europeo."
                segments.append(filler)

        document = "\n\n".join(segments)
        metadata["document_length"] = length
        metadata["num_events"] = len(events)

        return document, metadata

    def _generate_aggregation_document(self, length: int) -> Tuple[str, Dict]:
        """
        Genera documento per test 'Aggregation'.
        Dati numerici distribuiti che richiedono somme/calcoli.
        """
        segments = []
        metadata = {"transactions": [], "answers": {}}

        # Transazioni finanziarie distribuite
        transactions = [
            {"id": 1, "importo": 15000, "categoria": "vendite", "mese": "gennaio"},
            {"id": 2, "importo": 23000, "categoria": "vendite", "mese": "febbraio"},
            {"id": 3, "importo": 18500, "categoria": "vendite", "mese": "marzo"},
            {"id": 4, "importo": 31000, "categoria": "vendite", "mese": "aprile"},
            {"id": 5, "importo": 27500, "categoria": "vendite", "mese": "maggio"},
            {"id": 6, "importo": 42000, "categoria": "vendite", "mese": "giugno"},
            {"id": 7, "importo": 8000, "categoria": "spese", "mese": "gennaio"},
            {"id": 8, "importo": 12000, "categoria": "spese", "mese": "febbraio"},
            {"id": 9, "importo": 9500, "categoria": "spese", "mese": "marzo"},
            {"id": 10, "importo": 15000, "categoria": "spese", "mese": "aprile"},
            {"id": 11, "importo": 11000, "categoria": "spese", "mese": "maggio"},
            {"id": 12, "importo": 18000, "categoria": "spese", "mese": "giugno"},
        ]

        metadata["transactions"] = transactions

        # Calcoli corretti
        totale_vendite = sum(t["importo"] for t in transactions if t["categoria"] == "vendite")
        totale_spese = sum(t["importo"] for t in transactions if t["categoria"] == "spese")
        profitto = totale_vendite - totale_spese
        media_vendite_mensili = totale_vendite / 6
        mese_max_vendite = max((t for t in transactions if t["categoria"] == "vendite"), key=lambda x: x["importo"])

        metadata["answers"] = {
            "totale_vendite": totale_vendite,  # 157000
            "totale_spese": totale_spese,  # 73500
            "profitto": profitto,  # 83500
            "media_vendite_mensili": media_vendite_mensili,  # 26166.67
            "mese_max_vendite": mese_max_vendite["mese"],  # giugno
            "importo_max_vendite": mese_max_vendite["importo"],  # 42000
            "transazioni_totali": len(transactions),  # 12
        }

        # Genera segmenti
        for i in range(length):
            if i < len(transactions):
                t = transactions[i]
                segments.append(
                    f"[SEGMENTO {i}] REGISTRO FINANZIARIO #{t['id']}: "
                    f"{t['categoria'].capitalize()} per {t['mese']}. "
                    f"Importo: €{t['importo']:,.2f}. "
                    f"Transazione registrata nel sistema contabile."
                )
            else:
                filler = f"[SEGMENTO {i}] Nota amministrativa: Documentazione di supporto per audit e compliance."
                segments.append(filler)

        document = "\n\n".join(segments)
        metadata["document_length"] = length
        metadata["num_transactions"] = len(transactions)

        return document, metadata

    def _generate_entity_document(self, length: int) -> Tuple[str, Dict]:
        """
        Genera documento per test 'Entity Tracking'.
        Entità che appaiono in più segmenti con attributi che evolvono.
        """
        segments = []
        metadata = {"entities": {}, "answers": {}}

        # Entità con evoluzione nel tempo
        entities = {
            "Progetto Alpha": {
                "stato_iniziale": "pianificazione",
                "stato_finale": "completato",
                "budget_iniziale": 500000,
                "budget_finale": 650000,
                "team_iniziale": 5,
                "team_finale": 12,
                "menzioni": [0, 3, 7, 12, 18]
            },
            "Progetto Beta": {
                "stato_iniziale": "sviluppo",
                "stato_finale": "in pausa",
                "budget_iniziale": 300000,
                "budget_finale": 150000,
                "team_iniziale": 8,
                "team_finale": 2,
                "menzioni": [1, 5, 9, 14]
            },
            "Progetto Gamma": {
                "stato_iniziale": "ricerca",
                "stato_finale": "lancio_imminente",
                "budget_iniziale": 800000,
                "budget_finale": 1200000,
                "team_iniziale": 3,
                "team_finale": 15,
                "menzioni": [2, 6, 11, 16, 20]
            },
        }

        metadata["entities"] = entities

        # Domande e risposte
        metadata["answers"] = {
            "progetto_completato": "Progetto Alpha",
            "progetto_pausa": "Progetto Beta",
            "budget_maggiore": "Progetto Gamma",  # 1200000
            "team_cresciuto_di_piu": "Progetto Gamma",  # da 3 a 15 (+12)
            "progetti_totali": 3,
            "alpha_budget_aumento": 150000,  # 650000 - 500000
        }

        # Genera segmenti
        segment_index = 0
        for i in range(length):
            placed = False
            for entity_name, entity_data in entities.items():
                if i in entity_data["menzioni"] and segment_index < length:
                    menzione_idx = entity_data["menzioni"].index(i)
                    if menzione_idx == 0:
                        segments.append(
                            f"[SEGMENTO {i}] INIZIO PROGETTO: {entity_name} avviato. "
                            f"Stato: {entity_data['stato_iniziale']}. "
                            f"Budget allocato: €{entity_data['budget_iniziale']:,}. "
                            f"Team: {entity_data['team_iniziale']} membri."
                        )
                    elif menzione_idx == len(entity_data["menzioni"]) - 1:
                        segments.append(
                            f"[SEGMENTO {i}] STATO FINALE: {entity_name}. "
                            f"Stato attuale: {entity_data['stato_finale']}. "
                            f"Budget totale speso: €{entity_data['budget_finale']:,}. "
                            f"Team finale: {entity_data['team_finale']} membri."
                        )
                    else:
                        segments.append(
                            f"[SEGMENTO {i}] AGGIORNAMENTO: {entity_name} in corso. "
                            f"Milestone raggiunto. Risorse aggiuntive allocate."
                        )
                    placed = True
                    break

            if not placed:
                filler = f"[SEGMENTO {i}] Contesto generale: Informazioni operative sull'ambiente di progetto."
                segments.append(filler)

        document = "\n\n".join(segments)
        metadata["document_length"] = length

        return document, metadata

    def run_test(self, test_type: str, document_length: int = 50,
                 questions: List[str] = None) -> List[BenchmarkResult]:
        """
        Esegue un singolo tipo di test del benchmark.
        """
        print(f"\n{'='*60}")
        print(f"  TEST: {test_type.upper()}")
        print(f"  Lunghezza documento: {document_length} segmenti")
        print(f"{'='*60}")

        # Genera documento
        document, metadata = self.generate_test_document(test_type, document_length)

        # Ingerisci documento in BAS
        start_time = time.time()
        doc_id = self.bas.ingest_document(document, source_name=f"LOCOMO_{test_type}_{document_length}")
        ingestion_time = time.time() - start_time

        print(f"  Documento ingerito in {ingestion_time:.2f}s")
        print(f"  Agenti creati: {len(self.bas.segment_agents)}")

        # Seleziona domande da testare
        if questions is None:
            questions = list(metadata["answers"].keys())

        test_results = []

        for question_key in questions:
            if question_key not in metadata["answers"]:
                continue

            expected_answer = metadata["answers"][question_key]

            # Formula la domanda completa
            question = self._format_question(question_key, test_type)

            # Esegui query su BAS
            start_time = time.time()
            response = self.bas.query(
                question=question,
                max_rounds=3,
                countdown_seconds=60.0
            )
            latency = time.time() - start_time

            actual_answer = response.answer if hasattr(response, 'answer') else str(response)
            confidence = response.confidence if hasattr(response, 'confidence') else 0.0
            participating_agents = response.participating_agents if hasattr(response, 'participating_agents') else []

            # Valuta correttezza
            is_correct = self._evaluate_answer(actual_answer, expected_answer, test_type)

            # Calcola coerenza memoria
            memory_coherence = self.bas.get_memory_coherence()

            result = BenchmarkResult(
                test_name=f"{test_type}_{question_key}",
                test_type=test_type,
                document_length=document_length,
                question=question,
                expected_answer=str(expected_answer),
                actual_answer=actual_answer,
                is_correct=is_correct,
                confidence=confidence,
                latency_seconds=latency,
                agents_used=len(participating_agents),
                memory_coherence=memory_coherence,
                timestamp=datetime.now().isoformat()
            )

            test_results.append(result)
            self.results.append(result)

            # Stampa risultato
            status = "✓ PASS" if is_correct else "✗ FAIL"
            print(f"\n  {status} - {question_key}")
            print(f"    Domanda: {question[:80]}...")
            print(f"    Atteso: {expected_answer}")
            print(f"    Ottenuto: {actual_answer[:100]}...")
            print(f"    Confidence: {confidence:.2f}, Latency: {latency:.2f}s")

        return test_results

    def _format_question(self, question_key: str, test_type: str) -> str:
        """Formatta una domanda leggibile dalla chiave."""
        question_templates = {
            "needle": {
                "codice_segreto": "Qual è il codice segreto menzionato nel documento?",
                "nome_informatore": "Chi è l'informatore menzionato nel documento?",
                "luogo_incontro": "Dove deve avvenire l'incontro secondo il documento?",
                "password_accesso": "Qual è la password di accesso?",
                "numero_conto": "Qual è il numero di conto bancario indicato?",
                "data_lancio": "Quando è previsto il lancio?",
                "chiave_crittografia": "Quale chiave di crittografia viene utilizzata?",
                "contatto_emergenza": "Qual è il contatto di emergenza?",
            },
            "multi_hop": {
                "dove_lavora_marco": "Dove lavora Marco Rossi e in quale città si trova la sua azienda?",
                "abitanti_citta_marco": "Quanti abitanti ha la città dove lavora Marco Rossi?",
                "chi_ceo_datasolutions": "Chi è il CEO di DataSolutions Srl?",
                "anno_fondazione_innovazione": "In quale anno è stata fondata Innovazione SpA?",
                "sindaco_torino": "Chi è il sindaco di Torino?",
            },
            "temporal": {
                "primo_evento": "Qual è stato il primo evento nella timeline?",
                "ultimo_evento": "Qual è stato l'ultimo evento nella timeline?",
                "anni_durata": "Per quanti anni si estende la timeline degli eventi?",
                "acquisizione_gamma": "Quando Alpha ha acquisito Gamma Tech?",
                "eventi_2022": "Quanti eventi sono accaduti nel 2022?",
                "prima_del_2022": "Quanti eventi sono accaduti prima del 2022?",
                "dopo_acquisizione": "Cosa è successo dopo l'acquisizione di Gamma Tech?",
            },
            "aggregation": {
                "totale_vendite": "Qual è il totale delle vendite nei sei mesi?",
                "totale_spese": "Qual è il totale delle spese nei sei mesi?",
                "profitto": "Qual è il profitto totale (vendite - spese)?",
                "media_vendite_mensili": "Qual è la media mensile delle vendite?",
                "mese_max_vendite": "In quale mese le vendite sono state massime?",
                "importo_max_vendite": "Qual è l'importo massimo di vendite in un singolo mese?",
                "transazioni_totali": "Quante transazioni totali sono registrate?",
            },
            "entity": {
                "progetto_completato": "Quale progetto risulta completato?",
                "progetto_pausa": "Quale progetto è in pausa?",
                "budget_maggiore": "Quale progetto ha il budget finale maggiore?",
                "team_cresciuto_di_piu": "Quale progetto ha avuto la crescita maggiore del team?",
                "progetti_totali": "Quanti progetti sono menzionati nel documento?",
                "alpha_budget_aumento": "Di quanto è aumentato il budget del Progetto Alpha?",
            },
        }

        return question_templates.get(test_type, {}).get(question_key, f"Cos'è {question_key}?")

    def _evaluate_answer(self, actual: str, expected: Any, test_type: str) -> bool:
        """
        Valuta se la risposta è corretta.
        Implementa logica fuzzy per confronti numerici e testuali.
        """
        actual_lower = actual.lower().strip()
        expected_str = str(expected).lower().strip()

        # Confronto esatto per stringhe brevi
        if expected_str in actual_lower or actual_lower in expected_str:
            return True

        # Confronto numerico con tolleranza
        try:
            expected_num = float(expected)
            # Estrai numeri dalla risposta
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?|\d+(?:,\d{3})+', actual)
            for num_str in numbers:
                num_str = num_str.replace(',', '')
                actual_num = float(num_str)
                # Tolleranza del 5% per numeri grandi, 0.1 per piccoli
                tolerance = max(expected_num * 0.05, 0.1)
                if abs(actual_num - expected_num) <= tolerance:
                    return True
        except (ValueError, TypeError):
            pass

        # Parole chiave per tipi specifici
        if test_type == "needle":
            # Cerca corrispondenza parziale significativa
            expected_words = set(expected_str.split())
            actual_words = set(actual_lower.split())
            common_words = expected_words & actual_words
            if len(common_words) >= min(3, len(expected_words)):
                return True

        return False

    def run_full_benchmark(self, document_lengths: List[int] = [20, 50, 100]) -> BenchmarkSummary:
        """
        Esegue il benchmark completo su tutti i tipi di test e lunghezze.
        """
        print("\n" + "="*80)
        print("  LOCOMO BENCHMARK COMPLETO PER BAS")
        print("  Long Context Memory Benchmark")
        print("="*80)

        test_types = ["needle", "multi_hop", "temporal", "aggregation", "entity"]

        for length in document_lengths:
            for test_type in test_types:
                try:
                    self.run_test(test_type, document_length=length)
                except Exception as e:
                    print(f"  ERRORE in {test_type} ({length}): {e}")
                    # Crea risultato di errore
                    error_result = BenchmarkResult(
                        test_name=f"{test_type}_error_{length}",
                        test_type=test_type,
                        document_length=length,
                        question="ERROR",
                        expected_answer="N/A",
                        actual_answer=str(e),
                        is_correct=False,
                        confidence=0.0,
                        latency_seconds=0.0,
                        agents_used=0,
                        memory_coherence=0.0,
                        timestamp=datetime.now().isoformat()
                    )
                    self.results.append(error_result)

        return self.generate_summary()

    def generate_summary(self) -> BenchmarkSummary:
        """Genera il riepilogo statistico del benchmark."""
        if not self.results:
            return BenchmarkSummary(
                total_tests=0, passed=0, failed=0, accuracy=0.0,
                avg_latency=0.0, avg_agents_used=0.0, avg_memory_coherence=0.0,
                results_by_type={}, detailed_results=[],
                timestamp=datetime.now().isoformat()
            )

        total = len(self.results)
        passed = sum(1 for r in self.results if r.is_correct)
        failed = total - passed

        # Metriche aggregate
        avg_latency = sum(r.latency_seconds for r in self.results) / total
        avg_agents = sum(r.agents_used for r in self.results) / total
        avg_coherence = sum(r.memory_coherence for r in self.results) / total

        # Risultati per tipo
        results_by_type = {}
        for test_type in ["needle", "multi_hop", "temporal", "aggregation", "entity"]:
            type_results = [r for r in self.results if r.test_type == test_type]
            if type_results:
                type_passed = sum(1 for r in type_results if r.is_correct)
                results_by_type[test_type] = {
                    "total": len(type_results),
                    "passed": type_passed,
                    "accuracy": type_passed / len(type_results),
                    "avg_latency": sum(r.latency_seconds for r in type_results) / len(type_results),
                }

        summary = BenchmarkSummary(
            total_tests=total,
            passed=passed,
            failed=failed,
            accuracy=passed / total,
            avg_latency=avg_latency,
            avg_agents_used=avg_agents,
            avg_memory_coherence=avg_coherence,
            results_by_type=results_by_type,
            detailed_results=self.results,
            timestamp=datetime.now().isoformat()
        )

        return summary

    def print_summary(self, summary: BenchmarkSummary):
        """Stampa il riepilogo del benchmark in formato leggibile."""
        print("\n" + "="*80)
        print("  RISULTATI LOCOMO BENCHMARK")
        print("="*80)

        print(f"\n  TOTALE TEST: {summary.total_tests}")
        print(f"  ✓ PASSATI: {summary.passed}")
        print(f"  ✗ FALLITI: {summary.failed}")
        print(f"  ACCURATEZZA: {summary.accuracy:.2%}")

        print(f"\n  METRICHE PERFORMANCE:")
        print(f"    Latenza media: {summary.avg_latency:.2f}s")
        print(f"    Agenti usati (media): {summary.avg_agents_used:.1f}")
        print(f"    Coerenza memoria (media): {summary.avg_memory_coherence:.2f}")

        print(f"\n  RISULTATI PER TIPOLOGIA:")
        for test_type, metrics in summary.results_by_type.items():
            print(f"\n    {test_type.upper()}:")
            print(f"      Test: {metrics['total']}, Passati: {metrics['passed']}")
            print(f"      Accuratezza: {metrics['accuracy']:.2%}")
            print(f"      Latenza media: {metrics['avg_latency']:.2f}s")

        print("\n" + "="*80)


def main():
    """Esegue il benchmark LOCOMO completo."""

    # Inizializza BAS
    print("Inizializzazione BAS per LOCOMO Benchmark...")
    bas = BASEngine(
        max_agents=200,
        agents_per_query=10,
        countdown_seconds=30.0,
        judge_confidence_threshold=0.6,
        auto_god=True,
        dynamic_agent_selection=True
    )

    studsar = StudSarManager()

    # Crea benchmark runner
    benchmark = LOCOMOBenchmark(bas, studsar)

    # Esegui benchmark completo con diverse lunghezze
    summary = benchmark.run_full_benchmark(document_lengths=[20, 50])

    # Stampa risultati
    benchmark.print_summary(summary)

    # Salva risultati JSON
    output_file = "locomo_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total_tests": summary.total_tests,
                "passed": summary.passed,
                "failed": summary.failed,
                "accuracy": summary.accuracy,
                "avg_latency": summary.avg_latency,
                "avg_agents_used": summary.avg_agents_used,
                "avg_memory_coherence": summary.avg_memory_coherence,
                "results_by_type": summary.results_by_type,
            },
            "detailed_results": [asdict(r) for r in summary.detailed_results],
            "timestamp": summary.timestamp
        }, f, indent=2, default=str)

    print(f"\n  Risultati salvati in: {output_file}")

    return summary


if __name__ == "__main__":
    main()