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


class PostRetrievalExecutor:
    """
    Executes numerical reasoning and aggregation on retrieved segments.
    This module sits after retrieval and applies arithmetic/logic to answer aggregation queries.
    """
    
    def __init__(self):
        self.extracted_facts: List[NumericalFact] = []
        
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
        
        # Pattern for currency amounts with type marker
        # Matches: €15000.00 (income), €8500.00 (expense), etc.
        pattern = r'[\$€£]\s*([\d,.]+)\s*(million|billion|thousand|M|B|K)?\s*\((income|expense|entrata|uscita)\)?'
        
        for seg in segments:
            text = seg.get('text', '')
            seg_index = seg.get('index', 0)
            
            # Extract entities (simple heuristic: capitalized words)
            entity_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
            primary_entity = entity_matches[0] if entity_matches else "unknown"
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
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
        return max(filtered, key=lambda x: x.value)
    
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
        return min(filtered, key=lambda x: x.value)
    
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
        
        # Sum indicators
        if any(word in query_lower for word in ['total', 'sum', 'all', 'combined', 'altogether']):
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
                result = max_fact.value
                confidence = 0.75
                explanation = f"Maximum value: {result} (entity: {max_fact.entity})"
                
        elif op_type == 'min':
            min_fact = self.aggregate_min(facts)
            if min_fact:
                result = min_fact.value
                confidence = 0.75
                explanation = f"Minimum value: {result} (entity: {min_fact.entity})"
                
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
