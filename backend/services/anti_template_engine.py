"""
Anti-Template Engine for Mercury RFP System
Ensures uniqueness and prevents formulaic responses
"""

import hashlib
import json
import logging
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

class AntiTemplateEngine:
    """Prevents template-driven responses and ensures uniqueness"""

    def __init__(self):
        self.response_history_file = Path("data/response_history.jsonl")
        self.response_history_file.parent.mkdir(exist_ok=True)
        self.phrase_variations = self._load_phrase_variations()
        self.metaphor_banks = self._load_metaphor_banks()
        self.narrative_structures = self._load_narrative_structures()
        self.uniqueness_threshold = 0.7
        self.similarity_cache = {}

    def _load_phrase_variations(self) -> Dict[str, List[str]]:
        """Load variations for common phrases"""
        return {
            "we propose": [
                "our approach involves",
                "we recommend",
                "our solution encompasses",
                "we put forward",
                "our strategy includes"
            ],
            "will provide": [
                "delivers",
                "offers",
                "brings",
                "supplies",
                "furnishes"
            ],
            "innovative": [
                "cutting-edge",
                "pioneering",
                "forward-thinking",
                "advanced",
                "next-generation"
            ],
            "comprehensive": [
                "thorough",
                "complete",
                "extensive",
                "all-encompassing",
                "end-to-end"
            ],
            "leverage": [
                "utilize",
                "employ",
                "harness",
                "apply",
                "capitalize on"
            ],
            "demonstrate": [
                "show",
                "illustrate",
                "exhibit",
                "reveal",
                "evidence"
            ],
            "ensure": [
                "guarantee",
                "secure",
                "maintain",
                "safeguard",
                "establish"
            ]
        }

    def _load_metaphor_banks(self) -> Dict[str, List[str]]:
        """Load domain-specific metaphors"""
        return {
            "technical": [
                "building blocks",
                "foundation",
                "architecture",
                "framework",
                "ecosystem",
                "pipeline"
            ],
            "security": [
                "fortress",
                "shield",
                "guardian",
                "sentinel",
                "barrier",
                "safeguard"
            ],
            "innovation": [
                "catalyst",
                "accelerator",
                "springboard",
                "launchpad",
                "incubator",
                "forge"
            ],
            "collaboration": [
                "symphony",
                "tapestry",
                "mosaic",
                "constellation",
                "network",
                "web"
            ]
        }

    def _load_narrative_structures(self) -> List[str]:
        """Load different narrative approaches"""
        return [
            "problem_solution",
            "chronological",
            "comparative",
            "analytical",
            "case_study",
            "storytelling",
            "technical_deep_dive",
            "benefits_focused"
        ]

    async def ensure_uniqueness(
        self,
        content: str,
        document_type: str,
        rfp_context: Dict[str, Any]
    ) -> str:
        """Ensure content is unique and non-templated"""

        # Calculate current uniqueness score
        uniqueness_score = await self.calculate_uniqueness_score(content, document_type)

        logger.info(f"Initial uniqueness score: {uniqueness_score:.2f}")

        # If content is too similar to past responses, inject variations
        if uniqueness_score < self.uniqueness_threshold:
            content = await self.inject_variations(content, rfp_context)
            uniqueness_score = await self.calculate_uniqueness_score(content, document_type)
            logger.info(f"Post-variation uniqueness score: {uniqueness_score:.2f}")

        # Apply agency-specific adaptations
        content = await self.adapt_to_agency_culture(content, rfp_context)

        # Ensure narrative variety
        content = await self.vary_narrative_structure(content, document_type)

        # Record this response for future comparison
        await self.record_response(content, document_type, rfp_context)

        return content

    async def calculate_uniqueness_score(self, content: str, document_type: str) -> float:
        """Calculate how unique this content is compared to history"""

        # Get historical responses of same type
        historical = await self.get_historical_responses(document_type)

        if not historical:
            return 1.0  # First document of this type is maximally unique

        # Calculate similarity metrics
        similarities = []

        for past_content in historical[-10:]:  # Compare to last 10 responses
            similarity = self._calculate_text_similarity(content, past_content)
            similarities.append(similarity)

        # Uniqueness is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        uniqueness = 1.0 - avg_similarity

        return uniqueness

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""

        # Simple approach: Jaccard similarity of word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        jaccard_similarity = len(intersection) / len(union)

        # Also check for phrase similarity
        phrases1 = self._extract_phrases(text1)
        phrases2 = self._extract_phrases(text2)

        phrase_overlap = len(phrases1.intersection(phrases2)) / max(len(phrases1), len(phrases2), 1)

        # Weighted average
        return 0.6 * jaccard_similarity + 0.4 * phrase_overlap

    def _extract_phrases(self, text: str, n=3) -> set:
        """Extract n-gram phrases from text"""
        words = text.lower().split()
        phrases = set()

        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            phrases.add(phrase)

        return phrases

    async def inject_variations(self, content: str, rfp_context: Dict[str, Any]) -> str:
        """Inject variations to make content more unique"""

        # Replace common phrases with variations
        for phrase, variations in self.phrase_variations.items():
            if phrase in content.lower():
                replacement = random.choice(variations)
                content = re.sub(
                    re.escape(phrase),
                    replacement,
                    content,
                    flags=re.IGNORECASE,
                    count=1  # Only replace first occurrence
                )

        # Add context-specific metaphors
        content = await self.inject_metaphors(content, rfp_context)

        # Vary sentence structure
        content = self.vary_sentence_structure(content)

        # Add unique identifiers based on RFP
        content = self.add_rfp_specific_elements(content, rfp_context)

        return content

    async def inject_metaphors(self, content: str, rfp_context: Dict[str, Any]) -> str:
        """Add relevant metaphors based on context"""

        # Determine appropriate metaphor category
        if "security" in rfp_context.get('title', '').lower():
            metaphor_type = "security"
        elif "innovat" in rfp_context.get('title', '').lower():
            metaphor_type = "innovation"
        elif "collaborat" in rfp_context.get('title', '').lower():
            metaphor_type = "collaboration"
        else:
            metaphor_type = "technical"

        # Select and inject a metaphor
        if metaphor_type in self.metaphor_banks:
            metaphor = random.choice(self.metaphor_banks[metaphor_type])

            # Find a good place to insert the metaphor
            sentences = content.split('. ')
            if len(sentences) > 3:
                insert_point = random.randint(1, min(3, len(sentences)-1))
                metaphor_sentence = f"Like a {metaphor}, our solution provides the foundation for success"
                sentences.insert(insert_point, metaphor_sentence)
                content = '. '.join(sentences)

        return content

    def vary_sentence_structure(self, content: str) -> str:
        """Vary sentence structure to avoid monotony"""

        sentences = content.split('. ')
        modified_sentences = []

        for i, sentence in enumerate(sentences):
            if i % 3 == 0 and len(sentence) > 50:
                # Occasionally break long sentences
                if ',' in sentence:
                    parts = sentence.split(',', 1)
                    if len(parts) == 2:
                        modified_sentences.append(parts[0])
                        modified_sentences.append(parts[1].strip())
                    else:
                        modified_sentences.append(sentence)
                else:
                    modified_sentences.append(sentence)
            elif i % 5 == 0 and i > 0:
                # Occasionally combine short sentences
                if len(sentence) < 40 and i < len(sentences) - 1:
                    modified_sentences[-1] += f", and {sentence.lower()}"
                else:
                    modified_sentences.append(sentence)
            else:
                modified_sentences.append(sentence)

        return '. '.join(modified_sentences)

    def add_rfp_specific_elements(self, content: str, rfp_context: Dict[str, Any]) -> str:
        """Add RFP-specific elements to ensure uniqueness"""

        # Add agency-specific terminology if available
        agency = rfp_context.get('agency', '')
        if agency:
            agency_terms = self.get_agency_terminology(agency)
            if agency_terms:
                # Sprinkle agency-specific terms throughout
                term = random.choice(agency_terms)
                if term not in content:
                    # Add to introduction
                    content = content.replace(
                        "Our solution",
                        f"Our solution, aligned with {agency}'s {term},"
                    )

        # Add solicitation-specific reference
        sol_number = rfp_context.get('solicitation_number', '')
        if sol_number and sol_number not in content:
            content = f"In response to {sol_number}, {content}"

        return content

    def get_agency_terminology(self, agency: str) -> List[str]:
        """Get agency-specific terminology"""
        agency_terms = {
            "DOD": ["warfighter", "mission readiness", "force protection", "operational tempo"],
            "NASA": ["mission critical", "space heritage", "flight proven", "mission assurance"],
            "DHS": ["homeland security", "border protection", "critical infrastructure", "emergency response"],
            "GSA": ["federal marketplace", "acquisition excellence", "category management", "best value"],
            "VA": ["veteran-centric", "patient care", "healthcare delivery", "service excellence"]
        }

        for key in agency_terms:
            if key in agency.upper():
                return agency_terms[key]

        return []

    async def adapt_to_agency_culture(self, content: str, rfp_context: Dict[str, Any]) -> str:
        """Adapt content to specific agency culture and preferences"""

        agency = rfp_context.get('agency', '').upper()

        # Agency-specific adaptations
        if 'DOD' in agency or 'DEFENSE' in agency:
            # DoD prefers direct, mission-focused language
            content = self.make_more_direct(content)
            content = self.add_mission_focus(content)

        elif 'NASA' in agency:
            # NASA appreciates technical precision and innovation
            content = self.enhance_technical_precision(content)
            content = self.emphasize_innovation(content)

        elif 'VA' in agency or 'VETERAN' in agency:
            # VA values veteran-centric approach
            content = self.add_veteran_focus(content)

        elif 'DHS' in agency or 'HOMELAND' in agency:
            # DHS focuses on security and resilience
            content = self.emphasize_security(content)

        return content

    def make_more_direct(self, content: str) -> str:
        """Make language more direct for DoD"""
        replacements = {
            "we believe": "we will",
            "could potentially": "will",
            "might be able to": "can",
            "we think": "we know",
            "possibly": "",
            "perhaps": ""
        }

        for old, new in replacements.items():
            content = content.replace(old, new)

        return content

    def add_mission_focus(self, content: str) -> str:
        """Add mission-focused language"""
        if "mission" not in content.lower():
            content = content.replace(
                "Our solution",
                "Our mission-focused solution"
            )
        return content

    def enhance_technical_precision(self, content: str) -> str:
        """Enhance technical precision for NASA"""
        # Add precision qualifiers
        content = content.replace(
            "approximately",
            "precisely"
        )
        content = content.replace(
            "about",
            "exactly"
        )
        return content

    def emphasize_innovation(self, content: str) -> str:
        """Emphasize innovation for NASA"""
        if "innovat" not in content.lower():
            content = content.replace(
                "approach",
                "innovative approach",
                1  # Only first occurrence
            )
        return content

    def add_veteran_focus(self, content: str) -> str:
        """Add veteran-centric language for VA"""
        if "veteran" not in content.lower():
            content = content.replace(
                "users",
                "veterans",
                1
            )
        return content

    def emphasize_security(self, content: str) -> str:
        """Emphasize security for DHS"""
        if "security" not in content.lower()[:200]:  # Check first part
            content = content.replace(
                "Our solution",
                "Our secure solution"
            )
        return content

    async def vary_narrative_structure(self, content: str, document_type: str) -> str:
        """Vary the narrative structure of the content"""

        # Select a narrative structure based on document type and randomization
        if "technical" in document_type.lower():
            structures = ["technical_deep_dive", "problem_solution", "analytical"]
        elif "executive" in document_type.lower():
            structures = ["benefits_focused", "storytelling", "comparative"]
        else:
            structures = self.narrative_structures

        selected_structure = random.choice(structures)

        # Apply structure transformation
        if selected_structure == "problem_solution":
            content = self.structure_as_problem_solution(content)
        elif selected_structure == "benefits_focused":
            content = self.structure_as_benefits_focused(content)
        elif selected_structure == "storytelling":
            content = self.structure_as_story(content)

        return content

    def structure_as_problem_solution(self, content: str) -> str:
        """Structure content as problem-solution narrative"""
        if "Challenge:" not in content:
            # Add problem framing
            sentences = content.split('. ')
            if len(sentences) > 2:
                sentences[0] = f"Challenge: {sentences[0]}"
                sentences[2] = f"Solution: {sentences[2]}"
                content = '. '.join(sentences)
        return content

    def structure_as_benefits_focused(self, content: str) -> str:
        """Structure content with benefits focus"""
        if "Key Benefits:" not in content:
            # Add benefits framing
            sentences = content.split('. ')
            if len(sentences) > 3:
                sentences.insert(1, "Key Benefits: Enhanced efficiency, reduced costs, and improved outcomes")
                content = '. '.join(sentences)
        return content

    def structure_as_story(self, content: str) -> str:
        """Structure content as a story"""
        if not content.startswith("Imagine"):
            # Add storytelling element
            content = f"Imagine a solution that transforms your operations. {content}"
        return content

    async def get_historical_responses(self, document_type: str) -> List[str]:
        """Get historical responses for comparison"""
        historical = []

        if self.response_history_file.exists():
            with open(self.response_history_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('document_type') == document_type:
                            historical.append(record.get('content', ''))
                    except:
                        continue

        return historical

    async def record_response(self, content: str, document_type: str, rfp_context: Dict[str, Any]):
        """Record response for future uniqueness comparison"""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "document_type": document_type,
            "rfp_id": rfp_context.get('notice_id', 'unknown'),
            "agency": rfp_context.get('agency', ''),
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "content_length": len(content),
            "content": content[:1000]  # Store only first 1000 chars for comparison
        }

        with open(self.response_history_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    async def generate_uniqueness_report(self, content: str, document_type: str) -> Dict[str, Any]:
        """Generate a report on content uniqueness"""

        uniqueness_score = await self.calculate_uniqueness_score(content, document_type)
        historical = await self.get_historical_responses(document_type)

        # Analyze common phrases
        common_phrases = self.identify_common_phrases(content, historical)

        # Analyze vocabulary diversity
        vocab_diversity = self.calculate_vocabulary_diversity(content)

        # Identify overused terms
        overused_terms = self.identify_overused_terms(content)

        return {
            "uniqueness_score": uniqueness_score,
            "historical_comparisons": len(historical),
            "common_phrases": common_phrases,
            "vocabulary_diversity": vocab_diversity,
            "overused_terms": overused_terms,
            "recommendations": self.generate_uniqueness_recommendations(uniqueness_score)
        }

    def identify_common_phrases(self, content: str, historical: List[str]) -> List[str]:
        """Identify phrases common across responses"""
        current_phrases = self._extract_phrases(content)
        common = []

        for past_content in historical[-5:]:  # Check last 5
            past_phrases = self._extract_phrases(past_content)
            common.extend(list(current_phrases.intersection(past_phrases))[:3])

        return list(set(common))[:5]  # Top 5 common phrases

    def calculate_vocabulary_diversity(self, content: str) -> float:
        """Calculate vocabulary diversity score"""
        words = content.lower().split()
        unique_words = set(words)

        if not words:
            return 0.0

        # Type-token ratio
        diversity = len(unique_words) / len(words)
        return diversity

    def identify_overused_terms(self, content: str) -> List[Tuple[str, int]]:
        """Identify overused terms in content"""
        words = content.lower().split()
        word_counts = Counter(words)

        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                       'should', 'may', 'might', 'must', 'can', 'shall', 'that', 'this',
                       'these', 'those', 'our', 'we', 'us'}

        overused = []
        for word, count in word_counts.most_common(20):
            if word not in common_words and count > 3:
                overused.append((word, count))

        return overused[:5]

    def generate_uniqueness_recommendations(self, score: float) -> List[str]:
        """Generate recommendations to improve uniqueness"""
        recommendations = []

        if score < 0.5:
            recommendations.append("Content is highly templated - major restructuring needed")
            recommendations.append("Inject more RFP-specific details and context")
            recommendations.append("Vary narrative structure and sentence patterns")
        elif score < 0.7:
            recommendations.append("Add more unique perspectives and approaches")
            recommendations.append("Replace common phrases with varied expressions")
            recommendations.append("Include agency-specific adaptations")
        elif score < 0.85:
            recommendations.append("Good uniqueness - minor variations would help")
            recommendations.append("Consider adding distinctive metaphors or analogies")
        else:
            recommendations.append("Excellent uniqueness - content is highly distinctive")

        return recommendations