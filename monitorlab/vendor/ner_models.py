#!/usr/bin/env python3
"""
NER Models for Bergur Monitor
Contains dataclasses for NER entity recognition and consistency checking.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class NEREntity:
    """Represents a named entity with its properties"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    source: str = ""  # "rule_based" or "spacy"


@dataclass
class NERInconsistency:
    """Represents an inconsistency between rule-based and spaCy NER"""
    entity_text: str
    rule_based_label: str
    spacy_label: str
    rule_based_confidence: float
    spacy_confidence: float
    inconsistency_type: str  # "conflicting", "missing_in_spacy", "missing_in_rule_based"
    context: str = ""


@dataclass
class RuleDefinition:
    """Represents a rule definition for entity recognition"""
    name: str
    entity_type: str
    pattern: str
    description: str
    examples: List[str]
    confidence: float = 1.0
    is_regex: bool = True


@dataclass
class RuleValidationResult:
    """Result of validating a rule against text"""
    rule_name: str
    entity_type: str
    matches: List[Dict[str, Any]]
    total_matches: int
    confidence_score: float
    coverage_score: float  # How well the rule covers expected entities
    precision_score: float  # How precise the rule is (fewer false positives)
