"""
LOCOMO Benchmark per BAS - Versione Minimale Funzionante
Long Context Memory Benchmark (Meta AI Research, 2024)
"""

import sys
import time
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

# Aggiungi src al path
sys.path.insert(0, '/workspace')

from src.arena.bas_engine import BASEngine, BASResult


@dataclass
class BenchmarkResult:
    test_name: str
    test_type: str
    document_length: int
    question: str
    expected_answer: str
    actual_answer: str
    is_correct: bool
    confidence: float
    latency_seconds: float
    agents_used: int
    memory_coherence: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'test_type': self.test_type,
            'document_length': self.document_length,
            'question': self.question,
            'expected_answer': self.expected_answer,
            'actual_answer': self.actual_answer,
            'is_correct': self.is_correct,
            'confidence': self.confidence,
            'latency_seconds': self.latency_seconds,
            'agents_used': self.agents_used,
            'memory_coherence': self.memory_coherence,
            'timestamp': self.timestamp
        }


class LOCOMOBenchmark:
    """LOCOMO Benchmark per BAS."""
    
    def __init__(self):
        self.bas = BASEngine(max_agents=200, agents_per_query=10)
        self.results: List[BenchmarkResult] = []
        
    def _generate_needle_document(self, length: int) -> Tuple[str, Dict]:
        """Genera documento con 'aghi nascosti'."""
        segments = []
        metadata = {"needles": [], "answers": {}}
        
        # Nascondi 3-5 "aghi" in posizioni casuali
        needles = [
            ("informatore", "L'informatore chiave è Giovanni Rossi, contattabile tramite email criptata."),
            ("password", "La password master del sistema è: AlphaBeta1987!GammaDelta"),
            ("codice", "Il codice di accesso al vault è: 7394-XKCD-2841-MNPR"),
            ("appuntamento", "L'appuntamento segreto è fissato per il 15 marzo alle 23:45 al molo 7"),
            ("trasferimento", "Il trasferimento fondi da 2.5M€ è stato autorizzato il 3 febbraio 2024")
        ]
        
        needle_positions = set()
        import random
        while len(needle_positions) < min(5, length // 4):
            pos = random.randint(0, length - 1)
            needle_positions.add(pos)
        
        for i in range(length):
            if i in needle_positions and needles:
                needle_idx = len(needle_positions) - len(needle_positions) + list(needle_positions).index(i) % len(needles)
                key, text = needles[needle_idx % len(needles)]
                segments.append(f"[SEGMENTO {i}] {text}")
                metadata["needles"].append({"key": key, "position": i})
                metadata["answers"][key] = text
            else:
                filler = f"[SEGMENTO {i}] Documento amministrativo contenente informazioni di routine su procedure aziendali e protocolli operativi standard."
                segments.append(filler)
        
        document = "\n\n".join(segments)
        metadata["document_length"] = length
        return document, metadata
    
    def _generate_multi_hop_document(self, length: int) -> Tuple[str, Dict]:
        """Genera documento per multi-hop reasoning."""
        segments = []
        metadata = {"facts": {}, "answers": {}}
        
        entities = {
            "persona_1": {"nome": "Marco Rossi", "ruolo": "Direttore Tecnico", "azienda": "TechCorp Italia"},
            "persona_2": {"nome": "Laura Bianchi", "ruolo": "Responsabile Marketing", "azienda": "Innovazione SpA"},
            "azienda_1": {"nome": "TechCorp Italia", "settore": "Software", "citta": "Milano", "fondazione": 2010},
            "azienda_2": {"nome": "Innovazione SpA", "settore": "Consulenza", "citta": "Roma", "fondazione": 2015},
            "citta_1": {"nome": "Milano", "regione": "Lombardia", "abitanti": 1396000},
            "citta_2": {"nome": "Roma", "regione": "Lazio", "abitanti": 2873000},
        }
        
        metadata["facts"] = entities
        metadata["answers"] = {
            "dove_lavora_marco": "TechCorp Italia a Milano",
            "abitanti_citta_marco": 1396000,
            "chi_ceo_datasolutions": "Non presente nel documento",
            "anno_fondazione_innovazione": 2015,
        }
        
        facts_to_distribute = [
            ("persona_1", 0),
            ("azienda_1", 1),
            ("citta_1", 2),
            ("persona_2", 5),
            ("azienda_2", 6),
            ("citta_2", 7),
        ]
        
        for i in range(length):
            placed = False
            for entity_key, target_pos in facts_to_distribute:
                if i == target_pos and target_pos < length:
                    entity_data = entities.get(entity_key)
                    if entity_data is None:
                        continue
                    if "persona" in entity_key:
                        segments.append(f"[SEGMENTO {i}] Profile: {entity_data['nome']} ricopre il ruolo di {entity_data['ruolo']} presso {entity_data['azienda']}.")
                    elif "azienda" in entity_key:
                        segments.append(f"[SEGMENTO {i}] Azienda: {entity_data['nome']} opera nel settore {entity_data['settore']} con sede principale a {entity_data['citta']}. Fondata nel {entity_data['fondazione']}.")
                    elif "citta" in entity_key:
                        segments.append(f"[SEGMENTO {i}] Citta: {entity_data['nome']}, capoluogo della regione {entity_data['regione']}. Popolazione: {entity_data['abitanti']} abitanti.")
                    placed = True
                    break
            
            if not placed:
                filler_texts = [
                    f"[SEGMENTO {i}] Analisi di mercato: Il settore tecnologico mostra crescita costante.",
                    f"[SEGMENTO {i}] Report finanziario: Performance positive across all sectors.",
                    f"[SEGMENTO {i}] Nota operativa: Procedure standardizzate per tutti i dipartimenti.",
                ]
                segments.append(filler_texts[i % len(filler_texts)])
        
        document = "\n\n".join(segments)
        metadata["document_length"] = length
        return document, metadata
    
    def _generate_temporal_document(self, length: int) -> Tuple[str, Dict]:
        """Genera documento con timeline temporale."""
        segments = []
        metadata = {"events": [], "answers": {}}
        
        events = [
            {"data": "2020-01-15", "evento": "Fondazione startup Alpha", "tipo": "business"},
            {"data": "2020-06-20", "evento": "Alpha riceve seed funding", "tipo": "finance"},
            {"data": "2021-03-10", "evento": "Lancio prodotto flagship", "tipo": "product"},
            {"data": "2021-09-05", "evento": "Alpha acquisisce Gamma Tech", "tipo": "business"},
            {"data": "2022-02-28", "evento": "Gamma Tech diventa divisione di Alpha", "tipo": "business"},
            {"data": "2022-11-15", "evento": "Alpha supera 1 milione utenti", "tipo": "milestone"},
            {"data": "2023-04-22", "evento": "Alpha lancia versione 2.0", "tipo": "product"},
            {"data": "2024-01-10", "evento": "Alpha annuncia espansione", "tipo": "business"},
        ]
        
        metadata["events"] = events
        metadata["answers"] = {
            "quando_fondata": "2020-01-15",
            "primo_investimento": "2020-06-20",
            "quando_acquisizione": "2021-09-05",
            "quanti_anni_tra_fondazione_e_espansione": 4,
        }
        
        for i in range(length):
            if i < len(events):
                ev = events[i]
                segments.append(f"[SEGMENTO {i}] Data: {ev['data']}. Evento: {ev['evento']}. Tipo: {ev['tipo']}.")
            else:
                segments.append(f"[SEGMENTO {i}] Update periodico sulle operazioni aziendali in corso.")
        
        document = "\n\n".join(segments)
        metadata["document_length"] = length
        return document, metadata
    
    def _generate_aggregation_document(self, length: int) -> Tuple[str, Dict]:
        """Genera documento con dati numerici da aggregare."""
        segments = []
        metadata = {"transactions": [], "answers": {}}
        
        transactions = [
            {"id": 1, "importo": 15000, "tipo": "entrata", "descrizione": "Vendita prodotto A"},
            {"id": 2, "importo": 8500, "tipo": "uscita", "descrizione": "Acquisto materie prime"},
            {"id": 3, "importo": 22000, "tipo": "entrata", "descrizione": "Contratto cliente B"},
            {"id": 4, "importo": 5200, "tipo": "uscita", "descrizione": "Spese operative"},
            {"id": 5, "importo": 18000, "tipo": "entrata", "descrizione": "Servizi consulenza"},
            {"id": 6, "importo": 12000, "tipo": "uscita", "descrizione": "Stipendi personale"},
            {"id": 7, "importo": 9500, "tipo": "entrata", "descrizione": "Licenze software"},
            {"id": 8, "importo": 3200, "tipo": "uscita", "descrizione": "Utility e affitti"},
        ]
        
        total_entrate = sum(t["importo"] for t in transactions if t["tipo"] == "entrata")
        total_uscite = sum(t["importo"] for t in transactions if t["tipo"] == "uscita")
        
        metadata["transactions"] = transactions
        metadata["answers"] = {
            "totale_entrate": total_entrate,
            "totale_uscite": total_uscite,
            "bilancio_netto": total_entrate - total_uscite,
            "transazione_piu_grande": max(t["importo"] for t in transactions),
        }
        
        for i in range(length):
            if i < len(transactions):
                t = transactions[i]
                # Use English keywords for better filtering
                tipo_en = "income" if t["tipo"] == "entrata" else "expense"
                segments.append(f"[SEGMENT {i}] Transaction #{t['id']}: {t['descrizione']}. Amount: €{t['importo']:.2f} ({tipo_en}).")
            else:
                segments.append(f"[SEGMENT {i}] Accounting record: routine operation registered in the system.")
        
        document = "\n\n".join(segments)
        metadata["document_length"] = length
        return document, metadata
    
    def _generate_entity_document(self, length: int) -> Tuple[str, Dict]:
        """Genera documento con evoluzione entità."""
        segments = []
        metadata = {"entities": [], "answers": {}}
        
        projects = [
            {"nome": "Progetto Alpha", "fase_iniziale": "concept", "fase_finale": "lancio", "durata_mesi": 18},
            {"nome": "Progetto Beta", "fase_iniziale": "pianificazione", "fase_finale": "testing", "durata_mesi": 12},
            {"nome": "Progetto Gamma", "fase_iniziale": "sviluppo", "fase_finale": "deployment", "durata_mesi": 24},
        ]
        
        metadata["entities"] = projects
        metadata["answers"] = {
            "progetto_piu_lungo": "Progetto Gamma",
            "durata_media": sum(p["durata_mesi"] for p in projects) / len(projects),
            "quanti_progetti_hanno_raggiunto_lancio": 1,
        }
        
        for i in range(length):
            if i < len(projects):
                p = projects[i]
                segments.append(f"[SEGMENTO {i}] {p['nome']}: Fase iniziale={p['fase_iniziale']}, Fase finale={p['fase_finale']}, Durata={p['durata_mesi']} mesi.")
            else:
                segments.append(f"[SEGMENTO {i}] Report avanzamento progetti: aggiornamenti periodici sullo stato di avanzamento.")
        
        document = "\n\n".join(segments)
        metadata["document_length"] = length
        return document, metadata
    
    def _evaluate_answer(self, actual: str, expected: Any, test_type: str) -> bool:
        """Valuta se la risposta è corretta."""
        actual_lower = actual.lower()
        
        if test_type == "needle":
            # Cerca parole chiave dalla risposta attesa
            expected_str = str(expected).lower()
            parole_chiave = [p for p in expected_str.split() if len(p) > 4][:3]
            return any(p in actual_lower for p in parole_chiave)
        
        elif test_type == "multi_hop":
            expected_str = str(expected).lower()
            return expected_str in actual_lower or any(p in actual_lower for p in expected_str.split() if len(p) > 3)
        
        elif test_type == "temporal":
            # Per date o numeri
            if isinstance(expected, int):
                return str(expected) in actual
            expected_str = str(expected)
            return expected_str in actual_lower
        
        elif test_type == "aggregation":
            # Per numeri esatti
            if isinstance(expected, (int, float)):
                return str(int(expected)) in actual or f"{expected:.2f}" in actual
            return str(expected).lower() in actual_lower
        
        elif test_type == "entity":
            expected_str = str(expected).lower()
            if isinstance(expected, (int, float)):
                return str(int(expected)) in actual or str(expected) in actual
            return expected_str in actual_lower
        
        return False
    
    def run_test(self, test_type: str, length: int, questions: Dict[str, str], 
                 generate_fn, expected_answers: Dict[str, Any]) -> List[BenchmarkResult]:
        """Esegue un singolo test."""
        print(f"\n{'='*60}")
        print(f"  TEST: {test_type.upper()}")
        print(f"  Lunghezza documento: {length} segmenti")
        print(f"{'='*60}\n")
        
        # Genera documento
        document, metadata = generate_fn(length)
        
        # Ingerisci con BAS
        print(f"📥 BAS: Ingesting document 'LOCOMO_{test_type}_{length}'...")
        start_ingest = time.time()
        num_agents = self.bas.ingest_document(document, f"LOCOMO_{test_type}_{length}", segment_size=100)
        ingest_time = time.time() - start_ingest
        print(f"  ✓ Created {num_agents} agents in {ingest_time:.2f}s\n")
        
        test_results = []
        
        # Esegui domande
        for q_key, question in questions.items():
            print(f"**********************************************************************")
            print(f"  BAS QUERY: {question}")
            print(f"**********************************************************************")
            
            expected_answer = expected_answers.get(q_key, "Unknown")
            
            start_time = time.time()
            response = self.bas.query(question, max_rounds=3)
            latency = time.time() - start_time
            
            actual_answer = response.answer
            confidence = response.confidence
            participating_agents = response.participating_agents
            memory_coherence = self.bas.get_memory_coherence()
            
            is_correct = self._evaluate_answer(actual_answer, expected_answer, test_type)
            
            result = BenchmarkResult(
                test_name=f"{test_type}_{q_key}",
                test_type=test_type,
                document_length=length,
                question=question,
                expected_answer=str(expected_answer),
                actual_answer=actual_answer[:150] + "..." if len(actual_answer) > 150 else actual_answer,
                is_correct=is_correct,
                confidence=confidence,
                latency_seconds=latency,
                agents_used=len(participating_agents),
                memory_coherence=memory_coherence,
                timestamp=datetime.now().isoformat()
            )
            
            test_results.append(result)
            self.results.append(result)
            
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            print(f"  Expected: {expected_answer}")
            print(f"  Actual:   {actual_answer[:100]}...")
            print(f"  Confidence: {confidence:.3f}, Coherence: {memory_coherence:.3f}, Latency: {latency:.2f}s")
            print(f"  {status}\n")
        
        return test_results
    
    def run_full_benchmark(self):
        """Esegue il benchmark completo."""
        print("="*70)
        print("  LOCOMO BENCHMARK COMPLETO PER BAS")
        print("  Long Context Memory Benchmark")
        print("="*70)
        
        # Definizioni test
        tests = {
            "needle": {
                "questions": {
                    "informatore": "Chi è l'informatore menzionato nel documento?",
                    "password": "Qual è la password master del sistema?",
                    "codice": "Qual è il codice di accesso al vault?",
                },
                "lengths": [20, 50],
            },
            "multi_hop": {
                "questions": {
                    "dove_lavora_marco": "Dove lavora Marco Rossi e in quale città?",
                    "abitanti_citta_marco": "Quanti abitanti ha la città dove lavora Marco Rossi?",
                    "anno_fondazione_innovazione": "Quando è stata fondata Innovazione SpA?",
                },
                "lengths": [20, 50],
            },
            "temporal": {
                "questions": {
                    "quando_fondata": "Quando è stata fondata la startup Alpha?",
                    "primo_investimento": "Quando Alpha ha ricevuto il primo investimento?",
                    "quando_acquisizione": "Quando Alpha ha acquisito Gamma Tech?",
                },
                "lengths": [20, 50],
            },
            "aggregation": {
                "questions": {
                    "totale_entrate": "Qual è il totale delle entrate?",
                    "totale_uscite": "Qual è il totale delle uscite?",
                    "bilancio_netto": "Qual è il bilancio netto?",
                },
                "lengths": [20, 50],
            },
            "entity": {
                "questions": {
                    "progetto_piu_lungo": "Qual è il progetto più lungo?",
                    "durata_media": "Qual è la durata media dei progetti?",
                    "quanti_lancio": "Quanti progetti hanno raggiunto la fase di lancio?",
                },
                "lengths": [20, 50],
            },
        }
        
        # Esegui tutti i test
        for test_type, config in tests.items():
            generate_fn = getattr(self, f"_generate_{test_type}_document")
            
            for length in config["lengths"]:
                # Reset BAS per ogni test
                self.bas = BASEngine(max_agents=200, agents_per_query=10)
                
                # Ottieni answers dal documento generato
                _, metadata = generate_fn(length)
                expected_answers = metadata.get("answers", {})
                
                self.run_test(
                    test_type=test_type,
                    length=length,
                    questions=config["questions"],
                    generate_fn=generate_fn,
                    expected_answers=expected_answers
                )
        
        return self.results
    
    def print_summary(self):
        """Stampa riepilogo risultati."""
        print("\n" + "="*70)
        print("  RIEPILOGO RISULTATI LOCOMO BENCHMARK")
        print("="*70)
        
        if not self.results:
            print("  Nessun risultato disponibile.")
            return
        
        # Calcola statistiche
        total_tests = len(self.results)
        correct_tests = sum(1 for r in self.results if r.is_correct)
        accuracy = correct_tests / total_tests * 100 if total_tests > 0 else 0
        
        avg_confidence = sum(r.confidence for r in self.results) / total_tests
        avg_latency = sum(r.latency_seconds for r in self.results) / total_tests
        avg_coherence = sum(r.memory_coherence for r in self.results) / total_tests
        
        print(f"\n  Test totali:      {total_tests}")
        print(f"  Corretti:         {correct_tests}")
        print(f"  Errati:           {total_tests - correct_tests}")
        print(f"  Accuracy:         {accuracy:.1f}%")
        print(f"\n  Metriche medie:")
        print(f"    Confidence:     {avg_confidence:.3f}")
        print(f"    Coerenza Mem:   {avg_coherence:.3f}")
        print(f"    Latency:        {avg_latency:.2f}s")
        
        # Per categoria
        print(f"\n  Accuracy per categoria:")
        categories = {}
        for r in self.results:
            if r.test_type not in categories:
                categories[r.test_type] = {"total": 0, "correct": 0}
            categories[r.test_type]["total"] += 1
            if r.is_correct:
                categories[r.test_type]["correct"] += 1
        
        for cat, stats in sorted(categories.items()):
            cat_accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"    {cat:15s}: {cat_accuracy:5.1f}% ({stats['correct']}/{stats['total']})")
        
        # Salva risultati JSON
        output_file = "/workspace/locomo_benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "correct_tests": correct_tests,
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                    "avg_latency": avg_latency,
                    "avg_coherence": avg_coherence,
                    "timestamp": datetime.now().isoformat()
                },
                "by_category": {
                    cat: {
                        "accuracy": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0,
                        "correct": stats["correct"],
                        "total": stats["total"]
                    }
                    for cat, stats in categories.items()
                },
                "results": [r.to_dict() for r in self.results]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n  Risultati salvati in: {output_file}")
        print("="*70)


if __name__ == "__main__":
    benchmark = LOCOMOBenchmark()
    benchmark.run_full_benchmark()
    benchmark.print_summary()
