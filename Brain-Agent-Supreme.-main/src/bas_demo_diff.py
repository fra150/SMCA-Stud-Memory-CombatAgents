--- bas_demo.py (原始)


+++ bas_demo.py (修改后)
"""
BAS Demo — Brain Agent Supreme Demonstration

Questo script dimostra il funzionamento di BAS (Brain Agent Supreme),
il sistema dove N agenti = N segmenti di documento.

Principio: Documento → Segmenti → Agenti dedicati → Arena → Risposta Suprema
"""

import sys
import os

# Aggiungi src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.arena import BASEngine, SegmentAgent
from src.studsar import StudSarManager


def demo_bas_basic():
    """Demo base di BAS con un documento semplice."""

    print("\n" + "=" * 80)
    print("  BAS DEMO — Brain Agent Supreme")
    print("  Dimostrazione: N agenti = N segmenti di documento")
    print("=" * 80)

    # Crea motore BAS
    bas = BASEngine(
        max_agents=100,
        agents_per_query=5,
        countdown_seconds=30.0,
        judge_confidence_threshold=0.6,
        auto_god=True,
        dynamic_agent_selection=True
    )

    # Documento demo (semplificato per test rapido)
    document_text = """
    Artificial Intelligence is transforming the world in unprecedented ways.
    Machine learning algorithms can now recognize patterns in data that humans cannot see.
    Deep learning has revolutionized computer vision and natural language processing.
    Neural networks are inspired by the structure of the human brain.
    The future of AI holds immense promise for healthcare, education, and science.
    However, ethical considerations must guide AI development to ensure beneficial outcomes.
    Transparency and accountability are essential for trustworthy AI systems.
    Collaboration between humans and AI will define the next era of innovation.
    """

    # Ingerisci documento e crea agenti
    print("\n📥 Fase 1: Ingestione documento e creazione agenti")
    print("-" * 80)

    stats = bas.ingest_document(
        text=document_text,
        source_name="ai_intro",
        segment_length=50,  # Circa 50 parole per segmento
        create_agents=True
    )

    print(f"\n  Statistiche ingestione:")
    print(f"    Segmenti creati: {stats['segments_count']}")
    print(f"    Agenti creati: {stats['agents_created']}")
    print(f"    Totale agenti nel sistema: {stats['total_agents']}")
    print(f"    Marker StudSar: {stats['markers_after']}")

    # Query al sistema
    print("\n\n🔍 Fase 2: Query al sistema BAS")
    print("-" * 80)

    query = "What is the relationship between neural networks and the human brain?"

    result = bas.query(
        question=query,
        max_rounds=2,
        countdown_seconds=20.0
    )

    print(f"\n  Risultato BAS:")
    print(f"    Champion: {result.champion_name}")
    print(f"    Agenti attivi: {result.active_agents}/{result.total_segments}")
    print(f"    Confidenza: {result.judge_confidence:.3f}")
    print(f"    Coerenza memoria: {result.memory_coherence_score:.3f}")
    print(f"    Tempo: {result.total_processing_time:.2f}s")

    # Statistiche agenti
    print("\n\n📊 Fase 3: Statistiche agenti")
    print("-" * 80)

    agent_stats = bas.get_agent_statistics()
    print(f"\n  Statistiche generali:")
    print(f"    Totale agenti: {agent_stats['total_agents']}")
    print(f"    Attivazioni medie: {agent_stats['avg_activations']:.2f}")
    print(f"    Expertise media: {agent_stats['avg_expertise']:.3f}")

    if agent_stats.get('most_active_agents'):
        print(f"\n  Agenti più attivi:")
        for agent_id, activations in agent_stats['most_active_agents'][:3]:
            print(f"    - {agent_id}: {activations} attivazioni")

    # Salva stato
    print("\n\n💾 Fase 4: Salvataggio stato")
    print("-" * 80)

    saved_files = bas.save_state(directory="bas_demo_state")

    print(f"\n  Stato salvato con successo!")

    return bas, result


def demo_bas_multi_document():
    """Demo di BAS con multipli documenti."""

    print("\n" + "=" * 80)
    print("  BAS DEMO — Multi-Document")
    print("  Dimostrazione: Gestione di documenti multipli")
    print("=" * 80)

    # Crea motore BAS
    bas = BASEngine(
        max_agents=200,
        agents_per_query=8,
        countdown_seconds=30.0,
        dynamic_agent_selection=True
    )

    # Documento 1: Storia dell'AI
    doc1 = """
    The history of artificial intelligence began in the 1950s.
    Alan Turing proposed the famous Turing Test for machine intelligence.
    The Dartmouth Conference in 1956 coined the term 'artificial intelligence'.
    Early AI research focused on symbolic reasoning and problem solving.
    The first AI winter occurred in the 1970s due to limited computing power.
    Expert systems became popular in the 1980s for business applications.
    The second AI winter happened in the late 1980s and early 1990s.
    Machine learning emerged as a dominant paradigm in the 2000s.
    Deep learning breakthroughs started around 2012 with ImageNet.
    Today, AI is ubiquitous in smartphones, cars, and homes.
    """

    # Documento 2: Etica dell'AI
    doc2 = """
    AI ethics is crucial for ensuring beneficial technology development.
    Bias in AI systems can perpetuate societal inequalities.
    Transparency means understanding how AI makes decisions.
    Accountability requires clear responsibility for AI outcomes.
    Privacy concerns arise from AI's data collection capabilities.
    Fairness in AI requires diverse training data and testing.
    Human oversight remains essential for critical AI decisions.
    The precautionary principle suggests careful AI deployment.
    International cooperation is needed for AI governance.
    Ethical AI design should prioritize human wellbeing.
    """

    # Ingerisci entrambi i documenti
    print("\n📥 Ingestione documenti multipli...")

    stats1 = bas.ingest_document(doc1, source_name="ai_history", segment_length=40)
    stats2 = bas.ingest_document(doc2, source_name="ai_ethics", segment_length=40)

    print(f"\n  Documento 1 (History): {stats1['agents_created']} agenti")
    print(f"  Documento 2 (Ethics): {stats2['agents_created']} agenti")
    print(f"  Totale agenti: {len(bas.segment_agents)}")

    # Query che richiede conoscenza da entrambi i documenti
    query = "How has AI evolved historically and what ethical challenges emerged?"

    print(f"\n\n🔍 Query: {query}")
    print("-" * 80)

    result = bas.query(question=query, max_rounds=2)

    print(f"\n  Risultato:")
    print(f"    Agenti attivi: {result.active_agents}/{result.total_segments}")
    print(f"    Distribuzione segmenti: {result.segment_distribution}")
    print(f"    Coerenza: {result.memory_coherence_score:.3f}")

    return bas, result


def demo_comparison_smca_vs_bas():
    """Confronto concettuale tra SMCA e BAS."""

    print("\n" + "=" * 80)
    print("  CONFRONTO: SMCA vs BAS")
    print("=" * 80)

    comparison = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  CARATTERISTICA      │  SMCA              │  BAS                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Numero agenti       │  Fisso (es. 10)    │  Dinamico (N segmenti)│
    │  Specializzazione    │  Strategica        │  Per segmento/pagina  │
    │  Adattabilità        │  Limitata          │  Totale               │
    │  Documento 10 pag    │  10 agenti         │  10 agenti            │
    │  Documento 100 pag   │  10 agenti         │  100 agenti           │
    │  Documento 500 pag   │  10 agenti         │  500 agenti           │
    │  Memoria             │  Condivisa         │  Condivisa + specializzata│
    │  Selezione           │  Fissa             │  Dinamica per query   │
    │  Use case            │  Domini specifici  │  Documenti lunghi     │
    └─────────────────────────────────────────────────────────────────────┘

    IL SALTO CONCETTUALE:

    SMCA: Il sistema definisce l'arena (10 slot → adattati al documento)
    BAS:  Il documento definisce l'arena (N segmenti → N agenti)

    VANTAGGIO BAS:
    - Nessun LLM legge davvero 500 pagine — BAS sì, con 500 neuroni
    - Ogni agente è esperto del suo pezzo — conoscenza completa
    - StudSar connette tutto — cervello coerente, non frammenti
    """

    print(comparison)

    print("\n" + "=" * 80)
    print("  ROADMAP BAS")
    print("=" * 80)

    roadmap = """
    Fase 3 (ORA): BAS v1
      ✓ N agenti = N pagine documento
      ✓ StudSar centrale come memoria condivisa
      ✓ Selezione dinamica agenti per query
      ✓ Arena SMCA per selezione campione

    Fase 4 (PROSSIMO): BAS completo
      □ Auto-dimensionamento dinamico avanzato
      □ Pruning intelligente agenti inattivi
      □ Cross-agent learning (agenti imparano dagli altri)
      □ Risposta suprema multi-livello

    Fase 5 (FINALE): BAS supremo
      □ Autonomia quasi totale
      □ Dio interviene raramente
      □ Auto-miglioramento architetturale
      □ Scaling a migliaia di agenti
    """

    print(roadmap)


if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  BAS DEMO — Brain Agent Supreme".center(78) + "█")
    print("█" + "  Fase 4: Intelligenza Scalabile Suprema".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        # Demo 1: BAS base
        bas1, result1 = demo_bas_basic()

        # Demo 2: Multi-documento
        bas2, result2 = demo_bas_multi_document()

        # Confronto concettuale
        demo_comparison_smca_vs_bas()

        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  DEMO COMPLETATA CON SUCCESSO".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        print("\n✨ BAS è pronto per la Fase 4!")
        print("\nProssimi passi:")
        print("  1. Testare con documenti reali (500+ pagine)")
        print("  2. Ottimizzare selezione agenti per performance")
        print("  3. Implementare pruning dinamico agenti")
        print("  4. Aggiungere cross-agent learning")
        print("\n\"N pagine = N agenti = N neuroni dello stesso cervello\"\n")

    except Exception as e:
        print(f"\n❌ Errore durante la demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)