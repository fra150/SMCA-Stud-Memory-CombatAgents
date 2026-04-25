"""
Post-Retrieval Executor Module for BAS
Handles numerical reasoning, aggregation, and arithmetic operations on retrieved segments.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NumericalFact:
    """Represents a numerical fact extracted from text."""
    value: float
    unit: str
    entity: str
    context: str
    segment_index: int
    fact_type: Optional[str] = None  # 'income', 'expense', etc.

@dataclass
class TemporalFact:
    date: str
    entity: str
    event: str
    context: str
    segment_index: int


class PostRetrievalExecutor:
    """
    Executes numerical reasoning and aggregation on retrieved segments.
    This module sits after retrieval and applies arithmetic/logic to answer aggregation queries.
    """
    
    def __init__(self):
        self.extracted_facts: List[NumericalFact] = []
        self._entity_stopwords = {
            "durata", "fase", "segmento", "segment", "amount", "popolazione", "population",
            "transaction", "tipo", "evento", "data", "report", "profile", "azienda", "citta",
            "city", "capoluogo", "regione", "anno", "anni", "mese", "mesi", "giorno", "giorni",
            "ore", "ora", "minutes", "minute", "second", "seconds", "percent", "percentage",
            "totale", "media", "massimo", "minimo"
        }

    def _normalize_space(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _extract_candidate_entities(self, text: str) -> List[Tuple[str, int, int]]:
        if not text:
            return []

        tokens = list(re.finditer(r"[A-Za-zÀ-ÿ0-9]+", text))
        candidates: List[Tuple[str, int, int]] = []

        def is_upper_start(tok: str) -> bool:
            return bool(tok) and (tok[0].isupper() or tok[0] in "ÀÈÌÒÙÄÖÜ")

        def is_bridge_token(tok: str) -> bool:
            if not tok:
                return False
            if tok.lower() in self._entity_stopwords:
                return False
            if len(tok) <= 3 and tok.isupper():
                return True
            if tok.isdigit():
                return True
            return is_upper_start(tok)

        max_tokens = 5
        for i, m in enumerate(tokens):
            tok = m.group(0)
            if not is_upper_start(tok):
                continue
            if tok.lower() in self._entity_stopwords:
                continue

            start = m.start()
            end = m.end()
            last_end = end
            used = 1

            j = i + 1
            while j < len(tokens) and used < max_tokens:
                nxt = tokens[j].group(0)
                gap = tokens[j].start() - last_end
                if gap > 3:
                    break
                if not is_bridge_token(nxt):
                    break
                end = tokens[j].end()
                last_end = end
                used += 1
                j += 1

            ent = self._normalize_space(text[start:end])
            ent_l = ent.lower()
            if ent and ent_l not in self._entity_stopwords and len(ent) >= 2:
                candidates.append((ent, start, end))

        dedup = {}
        for ent, s, e in candidates:
            key = ent.lower()
            prev = dedup.get(key)
            if prev is None:
                dedup[key] = (ent, s, e)
            else:
                prev_ent, prev_s, prev_e = prev
                if (e - s) > (prev_e - prev_s):
                    dedup[key] = (ent, s, e)

        return list(dedup.values())

    def _choose_entity_for_match(self, text: str, match_start: int, match_end: int) -> str:
        candidates = self._extract_candidate_entities(text)
        if not candidates:
            return "unknown"

        window = 100
        preceding: List[Tuple[str, int, int, int]] = []
        following: List[Tuple[str, int, int, int]] = []

        for ent, s, e in candidates:
            if e <= match_start:
                dist = match_start - e
                if dist <= window:
                    preceding.append((ent, s, e, dist))
                continue

            if s >= match_end:
                dist = s - match_end
                if dist <= window:
                    following.append((ent, s, e, dist))
                continue

            preceding.append((ent, s, e, 0))

        def pick_best(items: List[Tuple[str, int, int, int]]) -> str:
            best = None
            for ent, s, e, dist in items:
                ent_len = e - s
                if best is None:
                    best = (dist, -ent_len, ent)
                    continue
                if (dist, -ent_len, ent) < best:
                    best = (dist, -ent_len, ent)
            return best[2] if best else "unknown"

        if preceding:
            return pick_best(preceding)
        if following:
            return pick_best(following)
        return "unknown"
        
    def extract_numerical_facts(self, segments: List[Dict[str, Any]], 
                                 filter_keywords: Optional[List[str]] = None) -> List[NumericalFact]:
        """
        Extract numerical facts from retrieved segments.
        
        Args:
            segments: List of segment dictionaries with 'text' and 'index' keys
            filter_keywords: Optional list of keywords to filter which numbers to extract
                            (e.g., ['income'] to only extract income values)
            
        Returns:
            List of extracted NumericalFact objects
        """
        facts = []
        
        # Pattern 1: currency amounts with type marker
        # Matches: €15000.00 (income), €8500.00 (expense), etc.
        pattern_currency = r'[\$€£]\s*([\d,.]+)\s*(million|billion|thousand|M|B|K)?\s*\((income|expense|entrata|uscita)\)?'
        
        # Pattern 2: general numbers with unit
        # Matches: Durata=24 mesi, 1396000 abitanti, ecc.
        pattern_general = r'(?:=|:\s*|^|\s)([\d,.]+)\s+(mesi|anni|giorni|ore|abitanti|utenti|percent|%)'
        
        for seg in segments:
            text = seg.get('text', '')
            seg_index = seg.get('index', 0)
            
            # 1. Match currencies
            matches_curr = re.finditer(pattern_currency, text, re.IGNORECASE)
            for match in matches_curr:
                try:
                    # Check if this fact matches the filter keywords
                    fact_type = match.group(3).lower() if match.lastindex >= 3 else None
                    if filter_keywords and fact_type and not any(k.lower() in fact_type for k in filter_keywords):
                        continue  # Skip this fact - doesn't match filter
                    
                    # Parse the number
                    num_str = match.group(1).replace(',', '')
                    value = float(num_str)
                    
                    # Handle multipliers
                    multiplier = 1
                    if match.lastindex >= 2 and match.group(2):
                        mult_str = match.group(2).lower()
                        if mult_str in ['million', 'm']:
                            multiplier = 1_000_000
                        elif mult_str in ['billion', 'b']:
                            multiplier = 1_000_000_000
                        elif mult_str in ['thousand', 'k']:
                            multiplier = 1_000
                    
                    final_value = value * multiplier
                    unit = match.group(2) if match.lastindex >= 2 else 'currency'
                    
                    # Get surrounding context
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    primary_entity = self._choose_entity_for_match(text, match.start(), match.end())
                        
                    fact = NumericalFact(
                        value=final_value,
                        unit=unit,
                        entity=primary_entity,
                        context=context,
                        segment_index=seg_index,
                        fact_type=fact_type
                    )
                    facts.append(fact)
                except (ValueError, IndexError):
                    continue

            # 2. Match general numbers and units
            matches_gen = re.finditer(pattern_general, text, re.IGNORECASE)
            for match in matches_gen:
                try:
                    unit = match.group(2).lower()
                    # Apply specific filter exceptions where currency-specific terms are requested
                    if filter_keywords and any(k in filter_keywords for k in ['income', 'expense', 'entrata', 'uscita', 'salary']):
                        continue # General numbers do not match specific financial queries
                        
                    num_str = match.group(1).replace(',', '')
                    value = float(num_str)
                    
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    primary_entity = self._choose_entity_for_match(text, match.start(), match.end())
                        
                    fact = NumericalFact(
                        value=value,
                        unit=unit,
                        entity=primary_entity,
                        context=context,
                        segment_index=seg_index,
                        fact_type=None
                    )
                    facts.append(fact)
                except (ValueError, IndexError):
                    continue
        
        self.extracted_facts = facts
        return facts
    
    def aggregate_sum(self, facts: List[NumericalFact], 
                      filter_entity: Optional[str] = None,
                      filter_unit: Optional[str] = None) -> Tuple[float, int]:
        """
        Sum numerical values matching optional filters.
        
        Args:
            facts: List of NumericalFact objects
            filter_entity: Only sum facts about this entity
            filter_unit: Only sum facts with this unit type
            
        Returns:
            Tuple of (sum, count of items summed)
        """
        filtered = facts
        if filter_entity:
            filtered = [f for f in filtered if filter_entity.lower() in f.entity.lower()]
        if filter_unit:
            filtered = [f for f in filtered if filter_unit.lower() in f.unit.lower()]
        
        total = sum(f.value for f in filtered)
        return total, len(filtered)
    
    def aggregate_count(self, facts: List[NumericalFact],
                       filter_entity: Optional[str] = None) -> int:
        """
        Count unique entities or facts.
        
        Args:
            facts: List of NumericalFact objects
            filter_entity: Only count facts about this entity
            
        Returns:
            Count of matching facts
        """
        filtered = facts
        if filter_entity:
            filtered = [f for f in filtered if filter_entity.lower() in f.entity.lower()]
        return len(filtered)
    
    def aggregate_max(self, facts: List[NumericalFact],
                     filter_unit: Optional[str] = None) -> Optional[NumericalFact]:
        """
        Find maximum value.
        
        Args:
            facts: List of NumericalFact objects
            filter_unit: Only consider facts with this unit type
            
        Returns:
            NumericalFact with maximum value, or None
        """
        filtered = facts
        if filter_unit:
            filtered = [f for f in filtered if filter_unit.lower() in f.unit.lower()]
        
        if not filtered:
            return None
        return max(filtered, key=lambda x: (x.value, -int(getattr(x, "segment_index", 0) or 0)))
    
    def aggregate_min(self, facts: List[NumericalFact],
                     filter_unit: Optional[str] = None) -> Optional[NumericalFact]:
        """
        Find minimum value.
        
        Args:
            facts: List of NumericalFact objects
            filter_unit: Only consider facts with this unit type
            
        Returns:
            NumericalFact with minimum value, or None
        """
        filtered = facts
        if filter_unit:
            filtered = [f for f in filtered if filter_unit.lower() in f.unit.lower()]
        
        if not filtered:
            return None
        return min(filtered, key=lambda x: (x.value, int(getattr(x, "segment_index", 0) or 0)))
    
    def filter_by_type(self, segments: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Filter segments by keywords based on query type before aggregation.
        
        Args:
            segments: List of segment dictionaries
            query: The user query
            
        Returns:
            Filtered list of segments matching the query type
        """
        query_lower = query.lower()
        
        # Determine keywords based on query type (support both Italian and English)
        # Key insight: Use specific English keywords only - "income" never matches "expense"
        if any(k in query_lower for k in ["entrata", "entrate", "income", "revenue", "earnings", "totale entrate", "budget", "net", "balance"]):
            # Income query: match ONLY "income" marker - automatically excludes "expense"
            keywords = ["income"]
        elif any(k in query_lower for k in ["uscita", "uscite", "expense", "expenditure", "cost", "spending", "totale uscite"]):
            # Expense query: match ONLY "expense" marker - automatically excludes "income"
            keywords = ["expense"]
        elif any(k in query_lower for k in ["salary", "wage", "compensation", "stipendio", "stipendi"]):
            keywords = ["salary", "wage", "compensation", "pay", "earned", "stipendio", "stipendi"]
        else:
            # No specific type mentioned, return all segments
            return segments
        
        # Filter segments that contain any of the keywords
        filtered = [
            s for s in segments
            if any(k in s.get('text', '').lower() for k in keywords)
        ]
        
        return filtered
    
    def detect_aggregation_query(self, query: str) -> Tuple[bool, str]:
        """
        Detect if a query requires aggregation and what type.
        
        Args:
            query: The user query
            
        Returns:
            Tuple of (is_aggregation, operation_type)
            operation_type can be: 'sum', 'count', 'max', 'min', 'average', or ''
        """
        query_lower = query.lower()
        
        # Superlative Patterns for direct intent mapping
        SUPERLATIVE_PATTERNS = {
            r"più (lungo|grande|alto|costoso|recente|vecchio|duraturo)": "max",
            r"più (corto|piccolo|basso|economico|veloce|breve)":      "min",
            r"men[oi] (lungo|grande|alto|costoso|recente|vecchio)":    "min",
            r"(massimo|maggiore|peggiore|migliore)":            "max",
            r"(minimo|minore)":                                 "min",
            r"quant[oi] .+ in totale":                         "sum",
            r"media":                                         "average"
        }
        
        for pattern, agg_type in SUPERLATIVE_PATTERNS.items():
            if re.search(pattern, query_lower):
                return True, agg_type
                
        # Sum indicators
        if any(word in query_lower for word in ['total', 'sum', 'all', 'combined', 'altogether', 'totale', 'bilancio', 'netto']):
            return True, 'sum'
        
        # Count indicators
        if any(word in query_lower for word in ['how many', 'count', 'number of', 'how much']):
            return True, 'count'
        
        # Max indicators
        if any(word in query_lower for word in ['largest', 'biggest', 'maximum', 'highest', 'most']):
            return True, 'max'
        
        # Min indicators
        if any(word in query_lower for word in ['smallest', 'minimum', 'lowest', 'least']):
            return True, 'min'
        
        # Average indicators
        if any(word in query_lower for word in ['average', 'mean', 'per']):
            return True, 'average'
        
        return False, ''
    
    def detect_temporal_query(self, query: str) -> bool:
        q = (query or "").strip().lower()
        if not q:
            return False
        if q.startswith("quando"):
            return True
        if "che data" in q or "in che data" in q or "in quale data" in q:
            return True
        if "data di" in q and ("evento" in q or "fondazione" in q):
            return True
        return False

    def extract_temporal_facts(self, segments: List[Dict[str, Any]]) -> List[TemporalFact]:
        facts: List[TemporalFact] = []
        date_pattern = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
        pair_pattern = re.compile(
            r"(?:Data|date)\s*:\s*(\d{4}-\d{2}-\d{2})[^\n]{0,180}?(?:Evento|event)\s*:\s*([^.\n]+)",
            re.IGNORECASE,
        )

        for seg in segments:
            text = seg.get("text", "") or ""
            seg_index = int(seg.get("index", 0) or 0)
            if not text:
                continue

            entity_candidates = sorted(self._extract_candidate_entities(text), key=lambda x: (x[2] - x[1]), reverse=True)
            primary_entity = entity_candidates[0][0] if entity_candidates else "unknown"

            covered_spans: List[Tuple[int, int]] = []
            for m in pair_pattern.finditer(text):
                date = m.group(1)
                event = (m.group(2) or "").strip()
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 40)
                context = self._normalize_space(text[start:end])
                facts.append(
                    TemporalFact(
                        date=date,
                        entity=primary_entity,
                        event=event,
                        context=context,
                        segment_index=seg_index,
                    )
                )
                covered_spans.append(m.span(1))

            for m in date_pattern.finditer(text):
                if any(s <= m.start() <= e for s, e in covered_spans):
                    continue
                date = m.group(1)
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                context = self._normalize_space(text[start:end])
                facts.append(
                    TemporalFact(
                        date=date,
                        entity=primary_entity,
                        event="",
                        context=context,
                        segment_index=seg_index,
                    )
                )
        return facts

    def execute_temporal(self, query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.detect_temporal_query(query):
            return {
                "result": None,
                "confidence": 0.0,
                "explanation": "Query does not require temporal extraction",
                "operation": None,
            }

        facts = self.extract_temporal_facts(segments)
        if not facts:
            return {
                "result": None,
                "confidence": 0.2,
                "explanation": "No ISO dates found in retrieved segments",
                "operation": "temporal",
            }

        q = (query or "").lower()
        q_entities = sorted(self._extract_candidate_entities(query or ""), key=lambda x: (x[2] - x[1]), reverse=True)
        entity_terms = [e for e, _, _ in q_entities if e and e.lower() not in self._entity_stopwords]

        event_patterns = [
            (re.compile(r"\bseed funding\b", re.IGNORECASE), 4.0),
            (re.compile(r"\bseries\s+[ab]\b", re.IGNORECASE), 3.0),
            (re.compile(r"partner\W+strategico", re.IGNORECASE), 3.0),
            (re.compile(r"\bacquis", re.IGNORECASE), 3.0),
            (re.compile(r"\bfond", re.IGNORECASE), 3.0),
            (re.compile(r"\blancio\b", re.IGNORECASE), 2.0),
            (re.compile(r"\bproduct\b", re.IGNORECASE), 1.0),
            (re.compile(r"\bfinance\b", re.IGNORECASE), 1.0),
            (re.compile(r"\bbusiness\b", re.IGNORECASE), 1.0),
        ]
        active_event_patterns = [pat for pat, _ in event_patterns if pat.search(q)]
        seg_by_idx = {int(s.get("index", 0) or 0): (s.get("text") or "") for s in segments}

        best = None
        best_score = float("-inf")

        for f in facts:
            score = 0.0
            hay = (f.context or "") + " " + (f.event or "")
            if active_event_patterns and not any(p.search(hay) for p in active_event_patterns):
                prev_txt = seg_by_idx.get(int(f.segment_index) - 1, "")
                next_txt = seg_by_idx.get(int(f.segment_index) + 1, "")
                hay = hay + " " + prev_txt + " " + next_txt

            matched_entity = False
            for ent in entity_terms[:3]:
                if ent and ent.lower() in hay.lower():
                    matched_entity = True
                    score += 3.0
            if entity_terms and not matched_entity:
                score -= 1.5

            for pat, w in event_patterns:
                if pat.search(q):
                    if pat.search(hay):
                        score += w
                    else:
                        score -= (w * 0.5)

            overlap = set(re.findall(r"[a-zàèìòù]+", q)) & set(re.findall(r"[a-zàèìòù]+", hay.lower()))
            score += min(2.0, len(overlap) * 0.2)

            if score > best_score:
                best_score = score
                best = f

        if not best:
            return {
                "result": None,
                "confidence": 0.2,
                "explanation": "Failed to rank temporal facts",
                "operation": "temporal",
            }

        confidence = max(0.35, min(0.9, 0.45 + (best_score / 10.0)))
        explanation = f"Matched date {best.date} (entity: {best.entity})"
        if best.event:
            explanation += f" event: {best.event}"

        return {
            "result": best.date,
            "confidence": confidence,
            "explanation": explanation,
            "operation": "temporal",
            "facts_used": len(facts),
        }

    def detect_multi_hop_query(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        return bool(
            re.search(r"^Dove lavora .+ e in quale citt", q, flags=re.IGNORECASE)
            or re.search(r"^Quanti abitanti ha la citt", q, flags=re.IGNORECASE)
            or re.search(r"^Quando è stata fondata ", q, flags=re.IGNORECASE)
        )

    def execute_multi_hop(self, query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.detect_multi_hop_query(query):
            return {
                "result": None,
                "confidence": 0.0,
                "explanation": "Query does not require multi-hop reasoning",
                "operation": None,
            }

        person_to_company: Dict[str, str] = {}
        company_to_city: Dict[str, str] = {}
        company_to_year: Dict[str, str] = {}
        city_to_population: Dict[str, str] = {}

        texts = []
        for seg in segments:
            t = seg.get("text", "") or ""
            if not t:
                continue
            t = re.sub(r"\[SEGMENTO\s+\d+\]\s*", " ", t, flags=re.IGNORECASE)
            texts.append(t)

        full_text = self._normalize_space(" ".join(texts))

        prof_block_pat = re.compile(r"Profile:\s*(.+?)(?=Profile:|Azienda:|Citta:|$)", re.IGNORECASE)
        for m in prof_block_pat.finditer(full_text):
            body = self._normalize_space(m.group(1))
            mm = re.search(r"^(?P<person>.+?)\s+ricopre il ruolo.+?presso\s+(?P<company>.+?)(?:\.|$)", body, flags=re.IGNORECASE)
            if not mm:
                continue
            person = self._normalize_space(mm.group("person"))
            company = self._normalize_space(mm.group("company"))
            if person and company:
                person_to_company[person] = company

        comp_block_pat = re.compile(r"Azienda:\s*(.+?)(?=Profile:|Azienda:|Citta:|$)", re.IGNORECASE)
        for m in comp_block_pat.finditer(full_text):
            body = self._normalize_space(m.group(1))
            m_company = re.search(r"^(?P<company>.+?)\s+opera nel settore", body, flags=re.IGNORECASE)
            m_city = re.search(r"sede principale a\s+(?P<city>.+?)(?:\.|Fondata nel|$)", body, flags=re.IGNORECASE)
            m_year = re.search(r"Fondata nel\s+(?P<year>\d{4})", body, flags=re.IGNORECASE)
            company = self._normalize_space(m_company.group("company")) if m_company else ""
            city = self._normalize_space(m_city.group("city")) if m_city else ""
            year = self._normalize_space(m_year.group("year")) if m_year else ""
            if company:
                if city:
                    company_to_city[company] = city
                if year:
                    company_to_year[company] = year

        city_block_pat = re.compile(r"Citta:\s*(.+?)(?=Profile:|Azienda:|Citta:|$)", re.IGNORECASE)
        for m in city_block_pat.finditer(full_text):
            body = self._normalize_space(m.group(1))
            m_city = re.search(r"^(?P<city>.+?)(?:,|\\s+capoluogo|\\.)", body, flags=re.IGNORECASE)
            m_pop = re.search(r"Popolazione:\s*(?P<pop>[0-9][0-9.,]*)\s+abitanti", body, flags=re.IGNORECASE)
            city = self._normalize_space(m_city.group("city")) if m_city else ""
            pop_raw = self._normalize_space(m_pop.group("pop")) if m_pop else ""
            pop = re.sub(r"[.,]", "", pop_raw) if pop_raw else ""
            if city and pop:
                city_to_population[city] = pop

        q = (query or "").strip()

        m1 = re.match(r"^Dove lavora\s+(.+?)\s+e in quale citt", q, flags=re.IGNORECASE)
        if m1:
            person = self._normalize_space(m1.group(1))
            company = person_to_company.get(person)
            city = company_to_city.get(company) if company else None
            if company and city:
                return {
                    "result": f"{company} a {city}",
                    "confidence": 0.9,
                    "explanation": "Chained person -> company -> city",
                    "operation": "multi_hop",
                }
            return {
                "result": None,
                "confidence": 0.3,
                "explanation": "Missing person/company/city link",
                "operation": "multi_hop",
            }

        m2 = re.match(r"^Quanti abitanti ha la citt.+dove lavora\s+(.+?)\?", q, flags=re.IGNORECASE)
        if m2:
            person = self._normalize_space(m2.group(1))
            company = person_to_company.get(person)
            city = company_to_city.get(company) if company else None
            pop = city_to_population.get(city) if city else None
            if pop:
                return {
                    "result": pop,
                    "confidence": 0.9,
                    "explanation": "Chained person -> company -> city -> population",
                    "operation": "multi_hop",
                }
            return {
                "result": None,
                "confidence": 0.3,
                "explanation": "Missing population link",
                "operation": "multi_hop",
            }

        m3 = re.match(r"^Quando è stata fondata\s+(.+?)\?", q, flags=re.IGNORECASE)
        if m3:
            company = self._normalize_space(m3.group(1))
            year = company_to_year.get(company)
            if year:
                return {
                    "result": year,
                    "confidence": 0.9,
                    "explanation": "Found company foundation year",
                    "operation": "multi_hop",
                }
            return {
                "result": None,
                "confidence": 0.3,
                "explanation": "Missing foundation year",
                "operation": "multi_hop",
            }

        return {
            "result": None,
            "confidence": 0.2,
            "explanation": "Unsupported multi-hop query pattern",
            "operation": "multi_hop",
        }

    def execute(self, query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute numerical reasoning on retrieved segments.
        
        Args:
            query: The user query
            segments: List of retrieved segments with 'text' and 'index' keys
            
        Returns:
            Dictionary with result, confidence, and explanation
        """
        # Detect if aggregation is needed
        is_agg, op_type = self.detect_aggregation_query(query)
        
        if not is_agg:
            return {
                'result': None,
                'confidence': 0.0,
                'explanation': 'Query does not require numerical aggregation',
                'operation': None
            }
        
        # Filter segments by type before extracting facts
        filtered_segments = self.filter_by_type(segments, query)
        
        # Determine filter keywords based on query type
        filter_keywords = None
        query_lower = query.lower()
        if any(k in query_lower for k in ["entrata", "entrate", "income", "revenue"]):
            filter_keywords = ['income', 'entrata']
        elif any(k in query_lower for k in ["uscita", "uscite", "expense", "expenditure"]):
            filter_keywords = ['expense', 'uscita']
        
        # Extract numerical facts from filtered segments with type filtering
        facts = self.extract_numerical_facts(filtered_segments, filter_keywords=filter_keywords)
        
        if not facts:
            return {
                'result': None,
                'confidence': 0.2,
                'explanation': 'No numerical facts found in retrieved segments after filtering',
                'operation': op_type,
                'segments_filtered': len(filtered_segments)
            }
        
        # Execute the appropriate operation
        result = None
        confidence = 0.5
        explanation = ""
        
        if op_type == 'sum':
            # Check if this is a net balance calculation (income - expense)
            if any(k in query.lower() for k in ["budget", "net", "balance", "saldo", "differenza"]):
                # Need both income and expense facts
                income_facts = self.extract_numerical_facts(segments, filter_keywords=['income', 'entrata'])
                expense_facts = self.extract_numerical_facts(segments, filter_keywords=['expense', 'uscita'])
                
                income_total = sum(f.value for f in income_facts)
                expense_total = sum(f.value for f in expense_facts)
                result = income_total - expense_total
                
                count = len(income_facts) + len(expense_facts)
                confidence = min(0.9, 0.5 + (count * 0.05))
                explanation = f"Net balance: income ({income_total}) - expense ({expense_total}) = {result}"
            else:
                total, count = self.aggregate_sum(facts)
                result = total
                confidence = min(0.9, 0.5 + (count * 0.05))
                explanation = f"Sum of {count} numerical values after filtering: {total}"
            
        elif op_type == 'count':
            count = self.aggregate_count(facts)
            result = count
            confidence = 0.8
            explanation = f"Counted {count} numerical facts"
            
        elif op_type == 'max':
            max_fact = self.aggregate_max(facts)
            if max_fact:
                result = max_fact.entity  # Fix: return entity name instead of numerical value
                confidence = 0.75
                explanation = f"Maximum value: {max_fact.value} (entity: {max_fact.entity})"
                
        elif op_type == 'min':
            min_fact = self.aggregate_min(facts)
            if min_fact:
                result = min_fact.entity  # Fix: return entity name instead of numerical value
                confidence = 0.75
                explanation = f"Minimum value: {min_fact.value} (entity: {min_fact.entity})"
                
        elif op_type == 'average':
            total, count = self.aggregate_sum(facts)
            if count > 0:
                result = total / count
                confidence = 0.7
                explanation = f"Average of {count} values: {result}"
        
        return {
            'result': result,
            'confidence': confidence,
            'explanation': explanation,
            'operation': op_type,
            'facts_used': len(facts),
            'segments_filtered': len(filtered_segments)
        }


# Convenience function for integration
def execute_numerical_reasoning(query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute numerical reasoning on retrieved segments.
    
    Args:
        query: User query
        segments: Retrieved segments
        
    Returns:
        Result dictionary
    """
    executor = PostRetrievalExecutor()
    return executor.execute(query, segments)


def execute_temporal_reasoning(query: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    executor = PostRetrievalExecutor()
    return executor.execute_temporal(query, segments)
