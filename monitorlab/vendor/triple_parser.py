#!/usr/bin/env python3
"""
Triple parsing utilities for Bergur Monitor ML.
Handles parsing of RDF triples from LLM responses.
"""

import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class TripleParser:
    """Handles parsing of RDF triples from LLM responses."""
    
    @staticmethod
    def parse_triples_from_text(text: str) -> List[Tuple[str, str, str]]:
        """
        Parse RDF triples from LLM response text with improved error handling.
        
        Args:
            text: Raw text containing RDF triples
            
        Returns:
            List of parsed triples as (subject, predicate, object) tuples
        """
        triples = []
        
        # Log the raw text for debugging
        # logger.info(f"Parsing triples from text (first 500 chars): {text[:500]}...")
        
        # Split text into lines and process each line
        lines = text.strip().split('\n')
        # logger.info(f"Found {len(lines)} lines to process")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # Log each line being processed
            logger.debug(f"Processing line {line_num}: {line}")
            
            # Try multiple parsing approaches
            parsed_triple = None
            
            # Approach 1: Standard format <subject, predicate, object>
            if line.startswith('<') and line.endswith('>'):
                parsed_triple = TripleParser._parse_angle_bracket_format(line)
            
            # Approach 2: No brackets, just comma-separated
            elif ',' in line and line.count(',') >= 2:
                parsed_triple = TripleParser._parse_comma_separated_format(line)
            
            # Approach 3: Space-separated triple
            elif ' ' in line:
                parsed_triple = TripleParser._parse_space_separated_format(line)
            
            if parsed_triple:
                triples.append(parsed_triple)
                logger.debug(f"Successfully parsed: {parsed_triple}")
            else:
                logger.warning(f"Could not parse line {line_num}: {line}")
        
        logger.info(f"Successfully parsed {len(triples)} triples from {len(lines)} lines")
        return triples
    
    @staticmethod
    def _parse_angle_bracket_format(line: str) -> Optional[Tuple[str, str, str]]:
        """Parse format: <subject, predicate, object>"""
        try:
            # Remove the outer < and > brackets
            content = line[1:-1]
            
            # Find the commas that separate the three parts
            comma_positions = []
            in_uri = False
            in_quotes = False
            
            for i, char in enumerate(content):
                if char == '"' and not in_uri:
                    in_quotes = not in_quotes
                elif char == '<' and not in_quotes:
                    in_uri = True
                elif char == '>' and not in_quotes:
                    in_uri = False
                elif char == ',' and not in_uri and not in_quotes:
                    comma_positions.append(i)
            
            # We need at least 2 commas to split into 3 parts
            if len(comma_positions) >= 2:
                first_comma = comma_positions[0]
                second_comma = comma_positions[1]
                
                subject = content[:first_comma].strip()
                predicate = content[first_comma + 1:second_comma].strip()
                obj = content[second_comma + 1:].strip()
                
                # Clean object - remove quotes if it's a literal
                obj = TripleParser._clean_object(obj)
                
                return (subject, predicate, obj)
        except Exception as e:
            logger.debug(f"Angle bracket parsing failed: {e}")
        return None
    
    @staticmethod
    def _parse_comma_separated_format(line: str) -> Optional[Tuple[str, str, str]]:
        """Parse format: subject, predicate, object (no brackets)"""
        try:
            parts = [part.strip() for part in line.split(',', 2)]
            if len(parts) >= 3:
                subject, predicate, obj = parts[0], parts[1], parts[2]
                obj = TripleParser._clean_object(obj)
                return (subject, predicate, obj)
        except Exception as e:
            logger.debug(f"Comma separated parsing failed: {e}")
        return None
    
    @staticmethod
    def _parse_space_separated_format(line: str) -> Optional[Tuple[str, str, str]]:
        """Parse format: subject predicate object (space-separated)"""
        try:
            # Handle URIs and quoted literals properly
            parts = []
            current_part = ""
            in_uri = False
            in_quotes = False
            
            for char in line:
                if char == '<' and not in_quotes:
                    in_uri = True
                    current_part += char
                elif char == '>' and not in_quotes:
                    in_uri = False
                    current_part += char
                elif char == '"' and not in_uri:
                    in_quotes = not in_quotes
                    current_part += char
                elif char == ' ' and not in_uri and not in_quotes:
                    if current_part.strip():
                        parts.append(current_part.strip())
                        current_part = ""
                else:
                    current_part += char
            
            # Add the last part
            if current_part.strip():
                parts.append(current_part.strip())
            
            if len(parts) >= 3:
                subject, predicate, obj = parts[0], parts[1], ' '.join(parts[2:])
                obj = TripleParser._clean_object(obj)
                return (subject, predicate, obj)
        except Exception as e:
            logger.debug(f"Space separated parsing failed: {e}")
        return None
    
    @staticmethod
    def _clean_object(obj: str) -> str:
        """Clean object string - remove quotes and language tags from literals"""
        try:
            # Handle quoted literals with language tags like "Berlin"@en
            if obj.startswith('"') and (obj.endswith('"') or '@' in obj):
                if '@' in obj:
                    obj = obj.split('@')[0][1:-1]  # Remove quotes and language tag
                elif obj.endswith('"'):
                    obj = obj[1:-1]  # Just remove quotes
            
            # Handle datatype literals like "2003"^^xsd:gYear
            elif '^^' in obj:
                obj = obj.split('^^')[0]
                if obj.startswith('"') and obj.endswith('"'):
                    obj = obj[1:-1]
        except Exception:
            pass  # Return original if cleaning fails
        
        return obj.strip()
