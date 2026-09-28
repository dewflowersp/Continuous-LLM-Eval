#!/usr/bin/env python3
"""
Metrics calculation utilities for Bergur Monitor ML.
Handles calculation of ICR, IPR, CI, and other metrics from RDF triples.
Also includes normalization utilities for scores.
"""

import networkx as nx
from typing import List, Tuple, Set, Dict, Any, Optional
import logging
import json
import re

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Handles calculation of various metrics from RDF triples."""
    
    @staticmethod
    def extract_unique_classes_from_triples(triples_list: List[Tuple[str, str, str]]) -> Set[str]:
        """
        Extract unique classes from the generated triples based on subjects and objects.
        
        Args:
            triples_list: List of (subject, predicate, object) tuples
            
        Returns:
            Set of unique class URIs
        """
        unique_classes = set()
        
        for subject, _, obj in triples_list:
            # Add subjects that are DBpedia ontology classes/resources
            if subject.startswith("http://dbpedia.org/"):
                unique_classes.add(subject)
            
            # Add objects that are DBpedia ontology classes/resources (not literals)
            # Exclude literal values (those with quotes, datatype markers, or language tags)
            if (obj.startswith("http://dbpedia.org/") and 
                not obj.startswith('"') and 
                '^^' not in obj and 
                '@' not in obj):
                unique_classes.add(obj)
        
        return unique_classes
    
    @staticmethod
    def find_instantiated_classes(triples_list: List[Tuple[str, str, str]], unique_classes: Set[str]) -> Set[str]:
        """
        Find classes that have instances (appear as objects in rdf:type relations).
        
        Args:
            triples_list: List of (subject, predicate, object) tuples
            unique_classes: Set of unique class URIs
            
        Returns:
            Set of instantiated class URIs
        """
        instantiated_classes = set()
        
        for _, predicate, obj in triples_list:
            # Look for rdf:type predicates or predicates containing 'type'
            if 'type' in predicate.lower() or 'rdf:type' in predicate:
                # Object is the class being instantiated
                if obj in unique_classes:
                    instantiated_classes.add(obj)
            
            # Also look for RDFS predicates that indicate class relationships
            if 'rdfs:' in predicate.lower() or 'rdfs' in predicate.lower():
                if obj in unique_classes:
                    instantiated_classes.add(obj)
        
        return instantiated_classes
    
    @staticmethod
    def calculate_icr_metric(triples_list: List[Tuple[str, str, str]]) -> Tuple[float, Set[str], Set[str]]:
        """
        Calculate ICR (Instantiated Class Ratio) metric.
        ICR = n(IC) / n(C)
        
        Args:
            triples_list: List of (subject, predicate, object) tuples
            
        Returns:
            Tuple of (ICR score, unique classes, instantiated classes)
        """
        # Step 1: Extract all unique classes from triples
        unique_classes = MetricsCalculator.extract_unique_classes_from_triples(triples_list)
        
        # Step 2: Find classes that have instances
        instantiated_classes = MetricsCalculator.find_instantiated_classes(triples_list, unique_classes)
        
        # Step 3: Calculate ICR
        n_C = len(unique_classes)  # Total number of classes
        n_IC = len(instantiated_classes)  # Number of instantiated classes
        
        if n_C == 0:
            icr = 0.0
        else:
            icr = n_IC / n_C
        
        return icr, unique_classes, instantiated_classes
    
    @staticmethod
    def extract_unique_properties_from_triples(triples_list: List[Tuple[str, str, str]]) -> Set[str]:
        """
        Extract unique properties (predicates) from the generated triples.
        
        Args:
            triples_list: List of (subject, predicate, object) tuples
            
        Returns:
            Set of unique property URIs
        """
        unique_properties = set()
        
        for _, predicate, _ in triples_list:
            # Add the predicate to unique properties
            unique_properties.add(predicate)
        
        return unique_properties
    
    @staticmethod
    def calculate_ipr_metric(triples_list: List[Tuple[str, str, str]]) -> Tuple[float, Set[str], Set[str]]:
        """
        Calculate IPR (Instantiated Property Ratio) metric.
        IPR = n(IP) / n(P)
        
        Args:
            triples_list: List of (subject, predicate, object) tuples
            
        Returns:
            Tuple of (IPR score, unique properties, instantiated properties)
        """
        # Step 1: Extract all unique properties from triples
        unique_properties = MetricsCalculator.extract_unique_properties_from_triples(triples_list)
        
        # Step 2: Count instantiated properties (properties from standard semantic web namespaces)
        instantiated_properties = set()
        
        # Standard semantic web namespaces
        standard_namespaces = [
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#',  # RDF
            'http://www.w3.org/2000/01/rdf-schema#',        # RDFS
            'http://www.w3.org/2002/07/owl#',               # OWL
            'http://xmlns.com/foaf/0.1/',                   # FOAF
            'http://purl.org/dc/elements/1.1/',             # Dublin Core
            'http://purl.org/dc/terms/',                    # Dublin Core Terms
        ]
        
        # Short form prefixes for standard namespaces
        standard_prefixes = ['rdf:', 'rdfs:', 'owl:', 'foaf:', 'dc:', 'dct:']
        
        for _, predicate, _ in triples_list:
            # Check for full URI form
            for namespace in standard_namespaces:
                if namespace in predicate:
                    instantiated_properties.add(predicate)
                    break
            else:
                # Check for short form (rdf:type, foaf:name, etc.)
                for prefix in standard_prefixes:
                    if predicate.startswith(prefix):
                        instantiated_properties.add(predicate)
                        break
        
        # Step 3: Calculate IPR
        n_P = len(unique_properties)  # Total number of unique properties
        n_IP = len(instantiated_properties)  # Number of standard namespace properties
        
        if n_P == 0:
            ipr = 0.0
        else:
            ipr = n_IP / n_P
        
        return ipr, unique_properties, instantiated_properties
    
    @staticmethod
    def calculate_ci(triples: str) -> float:
        """
        Calculate CI (Class Instantiation) from RDF triples.
        
        Args:
            triples: String containing RDF triples
            
        Returns:
            CI score
        """
        # Split the triples into lines
        lines = triples.strip().split("\n")
        # Count the number of unique classes
        unique_classes = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Each line is in the format: <subject, predicate, object>
            # Remove the angle brackets and split by comma
            if line.startswith('<') and line.endswith('>'):
                line = line[1:-1]  # Remove < and >
            parts = line.split(", ")
            if len(parts) >= 3:
                _, predicate, object_part = parts[0], parts[1], parts[2]
                if predicate == "rdf:type":
                    unique_classes.add(object_part)
        # Calculate CI as the ratio of instantiated classes to total classes
        if not unique_classes:
            return 0.0
        return len(unique_classes) / len(lines)

    
    @staticmethod
    def calculate_ci_user(triples: str) -> float:
        """
        Calculate CI (Class Instantiation) from RDF triples.

        Accepts either:
        - a JSON array of triples: [["sub","pred","obj"], ...]
        - a single-line JSON-encoded string containing that array
        - angle-bracket formatted lines: <sub, pred, obj>\n...

        Returns:
        CI score = (# unique classes from rdf:type) / (# total triples)
        """
        s = triples.strip()

        # 1) Try to extract and parse JSON content (handles double-encoded string too)
        parsed_triples = None
        try:
            # try direct JSON parse first
            parsed = json.loads(s)
            parsed_triples = parsed
        except Exception:
            # try to locate the first '[' and last ']' and parse that substring
            first = s.find('[')
            last = s.rfind(']')
            if first != -1 and last != -1 and last > first:
                candidate = s[first:last+1]
                try:
                    parsed_triples = json.loads(candidate)
                except Exception:
                    parsed_triples = None

        triples_list: List[List[str]] = []

        # If we successfully parsed JSON and it's a list-of-lists, normalize it
        if isinstance(parsed_triples, list):
            # sometimes the JSON is ["[ ... ]"] (one string) — try to unwrap that
            if len(parsed_triples) == 1 and isinstance(parsed_triples[0], str):
                try:
                    maybe_inner = json.loads(parsed_triples[0])
                    if isinstance(maybe_inner, list):
                        parsed_triples = maybe_inner
                except Exception:
                    pass

            # now expect parsed_triples to be list of triples (lists)
            for item in parsed_triples:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    subj, pred, obj = item[0], item[1], item[2]
                    triples_list.append([str(subj), str(pred), str(obj)])
        else:
            # Fallback: try to extract lines in angle-bracket format using regex
            # matches: <subject, predicate, object>
            pattern = re.compile(r'<\s*([^,>]+)\s*,\s*([^,>]+)\s*,\s*(.*?)\s*>')
            matches = pattern.findall(s)
            for m in matches:
                subj = m[0].strip()
                pred = m[1].strip()
                obj = m[2].strip().strip('"').strip("'")
                triples_list.append([subj, pred, obj])

        # If still empty, try to parse by splitting on '], [' as a last resort (handles your exact log)
        if not triples_list:
            # split entries like: ["s","p","o"], ["s2","p2","o2"]
            items = re.split(r'\]\s*,\s*\[', s.strip().lstrip('[').rstrip(']'))
            for it in items:
                parts = re.split(r'"\s*,\s*"', it.strip().strip('[]').strip())
                # remove surrounding quotes from parts
                parts = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
                if len(parts) >= 3:
                    triples_list.append([parts[0], parts[1], parts[2]])

        # Now compute CI
        unique_classes = set()
        total = 0
        for entry in triples_list:
            if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
                continue
            total += 1
            subj, pred, obj = entry[0], entry[1], entry[2]
            pred_clean = str(pred).strip().strip('"').strip("'")
            obj_clean = str(obj).strip().strip('"').strip("'")
            # normalize predicate (accept full URI or common short form)
            if pred_clean.endswith('rdf:type') or pred_clean == 'rdf:type':
                unique_classes.add(obj_clean)

        if total == 0:
            return 0.0

        return len(unique_classes) / total

    
    @staticmethod
    def create_knowledge_graph(entities: List[Tuple[str, str]], relationships: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """
        Create a NetworkX directed graph from entities and relationships.
        
        Args:
            entities: List of (text, label) tuples
            relationships: List of (subject, predicate, object) tuples
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes for each entity
        for entity, label in entities:
            G.add_node(entity, label=label)
        
        # Add edges for each relationship
        for subject, predicate, obj in relationships:
            if subject in G.nodes and obj in G.nodes:
                G.add_edge(subject, obj, relation=predicate)
        
        return G
    
    @staticmethod
    def calculate_ci_static_method(graph: nx.DiGraph) -> float:
        """
        Calculate CI (Connectivity Index) using graph-based method.
        CI = Number of connected components / Total number of nodes
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            CI score
        """
        if graph.number_of_nodes() == 0:
            return 1.0
        
        connected_components = list(nx.weakly_connected_components(graph))
        num_components = len(connected_components)
        ci = num_components / graph.number_of_nodes()
        
        return ci


# ============================================================================
# Normalization Utilities
# ============================================================================

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a single value using min-max normalization."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def normalize_scores_across_models(unified_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize scores across all models for a single news item.
    For each score item, the model with max value gets 1, min value gets 0.
    """
    if not unified_scores or len(unified_scores) == 0:
        return []
    
    # Extract all values for each metric across all models
    icr_values = [s.get("icr", 0) or 0 for s in unified_scores]
    ipr_values = [s.get("ipr", 0) or 0 for s in unified_scores]
    ci_values = [s.get("ci", 0) or 0 for s in unified_scores]
    latency_values = [s.get("latency", 0) or 0 for s in unified_scores]
    ner_consistency_values = [s.get("ner_consistency_score", 0) or 0 for s in unified_scores]
    rule_definition_values = [s.get("rule_definition_score", 0) or 0 for s in unified_scores]
    calculaner_tagging_values = [s.get("calculaner_tagging_consistency", 0) or 0 for s in unified_scores]
    cognitive_correctness_values = [s.get("cognitive_correctness", 0) or 0 for s in unified_scores]
    
    # Extract translation_correctness (can be number or object)
    translation_correctness_values = []
    for s in unified_scores:
        tc = s.get("translation_correctness")
        if isinstance(tc, (int, float)):
            translation_correctness_values.append(tc)
        elif isinstance(tc, dict):
            translation_correctness_values.append(tc.get("average_score", 0))
        else:
            translation_correctness_values.append(0)
    
    # Find min and max for each metric
    def get_min_max(values: List[float]) -> Dict[str, float]:
        filtered = [v for v in values if v is not None and not (isinstance(v, float) and (v != v))]  # Filter NaN
        if not filtered:
            return {"min": 0, "max": 0}
        return {"min": min(filtered), "max": max(filtered)}
    
    icr_min_max = get_min_max(icr_values)
    ipr_min_max = get_min_max(ipr_values)
    ci_min_max = get_min_max(ci_values)
    latency_min_max = get_min_max(latency_values)
    ner_consistency_min_max = get_min_max(ner_consistency_values)
    rule_definition_min_max = get_min_max(rule_definition_values)
    calculaner_tagging_min_max = get_min_max(calculaner_tagging_values)
    cognitive_correctness_min_max = get_min_max(cognitive_correctness_values)
    translation_correctness_min_max = get_min_max(translation_correctness_values)
    
    # Normalize each score
    normalized = []
    for i, score in enumerate(unified_scores):
        normalized_score = {
            "icr": normalize_value(icr_values[i], icr_min_max["min"], icr_min_max["max"]),
            "ipr": normalize_value(ipr_values[i], ipr_min_max["min"], ipr_min_max["max"]),
            "ci": normalize_value(ci_values[i], ci_min_max["min"], ci_min_max["max"]),
            "latency": normalize_value(latency_values[i], latency_min_max["min"], latency_min_max["max"]),
            "ner_consistency_score": normalize_value(
                ner_consistency_values[i],
                ner_consistency_min_max["min"],
                ner_consistency_min_max["max"]
            ),
            "rule_definition_score": normalize_value(
                rule_definition_values[i],
                rule_definition_min_max["min"],
                rule_definition_min_max["max"]
            ),
            "calculaner_tagging_consistency": normalize_value(
                calculaner_tagging_values[i],
                calculaner_tagging_min_max["min"],
                calculaner_tagging_min_max["max"]
            ),
        }
        
        # Optional fields
        if score.get("cognitive_correctness") is not None:
            normalized_score["cognitive_correctness"] = normalize_value(
                cognitive_correctness_values[i],
                cognitive_correctness_min_max["min"],
                cognitive_correctness_min_max["max"]
            )
        
        if score.get("translation_correctness") is not None:
            normalized_score["translation_correctness"] = normalize_value(
                translation_correctness_values[i],
                translation_correctness_min_max["min"],
                translation_correctness_min_max["max"]
            )
        
        normalized.append(normalized_score)
    
    return normalized


def min_max_normalize(values: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """Apply min-max normalization to an array of values."""
    if not values or len(values) == 0:
        return values
    
    data_min = min(values)
    data_max = max(values)
    
    if data_max == data_min:
        return values
    
    normalized = []
    for value in values:
        normalized_value = ((value - data_min) / (data_max - data_min)) * (max_val - min_val) + min_val
        normalized.append(normalized_value)
    
    return normalized


def normalize_model_data(model_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize sensitivity and sentiment scores in the model data.
    Core metrics (icr, ipr, ci, etc.) are already normalized per news item and should not be re-normalized.
    """
    if not model_data or len(model_data) == 0:
        return model_data
    
    # Collect only sensitivity and sentiment score values across all models for normalization
    all_scores = {
        "sensitivity_race_ethnicity": [],
        "sensitivity_religion": [],
        "sensitivity_gender_sexuality": [],
        "sensitivity_geography": [],
        "sensitivity_socioeconomic": [],
        "sensitivity_political_orientation": [],
        "sentiment_race_ethnicity": [],
        "sentiment_religion": [],
        "sentiment_gender_sexuality": [],
        "sentiment_geography": [],
        "sentiment_socioeconomic": [],
        "sentiment_political_orientation": [],
    }
    
    # Collect sensitivity and sentiment values across all models and data points
    for model in model_data:
        for item in model.get("data", []):
            # Collect sensitivity scores
            if item.get("sensitivity_score"):
                ss = item["sensitivity_score"]
                all_scores["sensitivity_race_ethnicity"].append(ss.get("race_ethnicity_analysis", 0))
                all_scores["sensitivity_religion"].append(ss.get("religion_analysis", 0))
                all_scores["sensitivity_gender_sexuality"].append(ss.get("gender_sexuality_analysis", 0))
                all_scores["sensitivity_geography"].append(ss.get("geography_analysis", 0))
                all_scores["sensitivity_socioeconomic"].append(ss.get("socioeconomic_analysis", 0))
                all_scores["sensitivity_political_orientation"].append(ss.get("political_orientation_analysis", 0))
            
            # Collect sentiment scores
            if item.get("sentiment_score"):
                sts = item["sentiment_score"]
                all_scores["sentiment_race_ethnicity"].append(sts.get("race_ethnicity_analysis", 0))
                all_scores["sentiment_religion"].append(sts.get("religion_analysis", 0))
                all_scores["sentiment_gender_sexuality"].append(sts.get("gender_sexuality_analysis", 0))
                all_scores["sentiment_geography"].append(sts.get("geography_analysis", 0))
                all_scores["sentiment_socioeconomic"].append(sts.get("socioeconomic_analysis", 0))
                all_scores["sentiment_political_orientation"].append(sts.get("political_orientation_analysis", 0))
    
    # Normalize only sensitivity and sentiment scores
    normalized = {
        "sensitivity_race_ethnicity": min_max_normalize(all_scores["sensitivity_race_ethnicity"]),
        "sensitivity_religion": min_max_normalize(all_scores["sensitivity_religion"]),
        "sensitivity_gender_sexuality": min_max_normalize(all_scores["sensitivity_gender_sexuality"]),
        "sensitivity_geography": min_max_normalize(all_scores["sensitivity_geography"]),
        "sensitivity_socioeconomic": min_max_normalize(all_scores["sensitivity_socioeconomic"]),
        "sensitivity_political_orientation": min_max_normalize(all_scores["sensitivity_political_orientation"]),
        "sentiment_race_ethnicity": min_max_normalize(all_scores["sentiment_race_ethnicity"]),
        "sentiment_religion": min_max_normalize(all_scores["sentiment_religion"]),
        "sentiment_gender_sexuality": min_max_normalize(all_scores["sentiment_gender_sexuality"]),
        "sentiment_geography": min_max_normalize(all_scores["sentiment_geography"]),
        "sentiment_socioeconomic": min_max_normalize(all_scores["sentiment_socioeconomic"]),
        "sentiment_political_orientation": min_max_normalize(all_scores["sentiment_political_orientation"]),
    }
    
    # Apply normalized values back to the model data (only sensitivity and sentiment)
    index = 0
    normalized_model_data = []
    for model in model_data:
        normalized_model = {
            **model,
            "data": []
        }
        for item in model.get("data", []):
            normalized_item = {
                **item,  # Keep all existing values including already-normalized core metrics
                # Only update sensitivity and sentiment scores
                "sensitivity_score": {
                    "race_ethnicity_analysis": normalized["sensitivity_race_ethnicity"][index] if index < len(normalized["sensitivity_race_ethnicity"]) else 0,
                    "religion_analysis": normalized["sensitivity_religion"][index] if index < len(normalized["sensitivity_religion"]) else 0,
                    "gender_sexuality_analysis": normalized["sensitivity_gender_sexuality"][index] if index < len(normalized["sensitivity_gender_sexuality"]) else 0,
                    "geography_analysis": normalized["sensitivity_geography"][index] if index < len(normalized["sensitivity_geography"]) else 0,
                    "socioeconomic_analysis": normalized["sensitivity_socioeconomic"][index] if index < len(normalized["sensitivity_socioeconomic"]) else 0,
                    "political_orientation_analysis": normalized["sensitivity_political_orientation"][index] if index < len(normalized["sensitivity_political_orientation"]) else 0,
                },
                "sentiment_score": {
                    "race_ethnicity_analysis": normalized["sentiment_race_ethnicity"][index] if index < len(normalized["sentiment_race_ethnicity"]) else 0,
                    "religion_analysis": normalized["sentiment_religion"][index] if index < len(normalized["sentiment_religion"]) else 0,
                    "gender_sexuality_analysis": normalized["sentiment_gender_sexuality"][index] if index < len(normalized["sentiment_gender_sexuality"]) else 0,
                    "geography_analysis": normalized["sentiment_geography"][index] if index < len(normalized["sentiment_geography"]) else 0,
                    "socioeconomic_analysis": normalized["sentiment_socioeconomic"][index] if index < len(normalized["sentiment_socioeconomic"]) else 0,
                    "political_orientation_analysis": normalized["sentiment_political_orientation"][index] if index < len(normalized["sentiment_political_orientation"]) else 0,
                }
            }
            normalized_model["data"].append(normalized_item)
            index += 1
        normalized_model_data.append(normalized_model)
    
    return normalized_model_data
