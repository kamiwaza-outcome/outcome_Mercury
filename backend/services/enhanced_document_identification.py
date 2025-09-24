"""
Enhanced Document Identification System for Multi-Agent Orchestrator
Replaces primitive keyword matching with GPT-5 intelligent analysis
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Federal document type classifications"""
    TECHNICAL_VOLUME = "technical_volume"
    COST_VOLUME = "cost_volume"
    MANAGEMENT_VOLUME = "management_volume"
    PAST_PERFORMANCE = "past_performance"
    COVER_LETTER = "cover_letter"
    EXECUTIVE_SUMMARY = "executive_summary"
    PRICE_NARRATIVE = "price_narrative"
    SOW_RESPONSE = "sow_response"
    TECHNICAL_PROPOSAL = "technical_proposal"
    COMPLIANCE_MATRIX = "compliance_matrix"
    SECURITY_PLAN = "security_plan"
    STAFFING_PLAN = "staffing_plan"
    CAPABILITY_STATEMENT = "capability_statement"
    VIDEO_PITCH = "video_pitch"
    WHITE_PAPER = "white_paper"
    ADDENDUM = "addendum"
    APPENDIX = "appendix"


@dataclass
class DocumentPattern:
    """Pattern definition for document recognition"""
    document_type: DocumentType
    keywords: List[str]
    phrases: List[str]
    volume_indicators: List[str]
    federal_conventions: List[str]
    confidence_weights: Dict[str, float]
    format_indicators: List[str]


@dataclass
class VolumeStructure:
    """Volume structure detection"""
    has_volumes: bool
    volume_count: int
    volume_mapping: Dict[str, str]  # Volume number -> content type
    numbering_style: str  # "roman", "numeric", "alpha"


@dataclass
class DocumentIdentification:
    """Result of document identification"""
    name: str
    document_type: DocumentType
    requirements: str
    format: str
    confidence_score: float
    reasoning: str
    volume_info: Optional[VolumeStructure]
    page_limit: Optional[int]
    specialized: bool
    consolidates: List[str]
    assumptions: List[str]


class FederalDocumentRegistry:
    """Comprehensive registry of federal document patterns"""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.volume_patterns = self._initialize_volume_patterns()
        self.federal_conventions = self._initialize_federal_conventions()

    def _initialize_patterns(self) -> Dict[DocumentType, DocumentPattern]:
        """Initialize comprehensive document patterns"""
        return {
            DocumentType.TECHNICAL_VOLUME: DocumentPattern(
                document_type=DocumentType.TECHNICAL_VOLUME,
                keywords=[
                    "technical", "architecture", "approach", "solution", "design",
                    "implementation", "methodology", "system", "engineering",
                    "integration", "development", "technology"
                ],
                phrases=[
                    "technical volume", "volume i", "volume 1", "technical approach",
                    "technical solution", "system architecture", "technical proposal",
                    "implementation plan", "technical requirements", "solution architecture"
                ],
                volume_indicators=["volume i", "volume 1", "vol i", "vol 1", "volume one"],
                federal_conventions=[
                    "performance work statement response", "pws response",
                    "statement of work response", "sow response",
                    "technical requirements compliance", "factor 1"
                ],
                confidence_weights={
                    "volume_indicator": 0.4,
                    "keywords": 0.3,
                    "phrases": 0.2,
                    "federal_convention": 0.1
                },
                format_indicators=["docx", "pdf", "doc"]
            ),

            DocumentType.COST_VOLUME: DocumentPattern(
                document_type=DocumentType.COST_VOLUME,
                keywords=[
                    "cost", "price", "pricing", "budget", "financial", "rates",
                    "billing", "fee", "proposal", "economic", "value"
                ],
                phrases=[
                    "cost volume", "volume ii", "volume 2", "cost proposal",
                    "price proposal", "cost breakdown", "pricing structure",
                    "cost narrative", "cost summary", "budget proposal"
                ],
                volume_indicators=["volume ii", "volume 2", "vol ii", "vol 2", "volume two"],
                federal_conventions=[
                    "cost or pricing data", "far part 31", "cost principles",
                    "indirect rates", "direct costs", "factor 2", "cost realism"
                ],
                confidence_weights={
                    "volume_indicator": 0.4,
                    "keywords": 0.3,
                    "phrases": 0.2,
                    "federal_convention": 0.1
                },
                format_indicators=["xlsx", "xls", "pdf", "docx"]
            ),

            DocumentType.MANAGEMENT_VOLUME: DocumentPattern(
                document_type=DocumentType.MANAGEMENT_VOLUME,
                keywords=[
                    "management", "organization", "staffing", "personnel", "team",
                    "project", "schedule", "timeline", "resources", "roles"
                ],
                phrases=[
                    "management volume", "volume iii", "volume 3", "management approach",
                    "organizational structure", "project management", "staffing plan",
                    "management proposal", "organizational chart"
                ],
                volume_indicators=["volume iii", "volume 3", "vol iii", "vol 3", "volume three"],
                federal_conventions=[
                    "project management plan", "pmp", "organizational capabilities",
                    "key personnel", "factor 3", "management approach"
                ],
                confidence_weights={
                    "volume_indicator": 0.4,
                    "keywords": 0.3,
                    "phrases": 0.2,
                    "federal_convention": 0.1
                },
                format_indicators=["docx", "pdf", "doc"]
            ),

            DocumentType.PAST_PERFORMANCE: DocumentPattern(
                document_type=DocumentType.PAST_PERFORMANCE,
                keywords=[
                    "past", "performance", "experience", "history", "previous",
                    "prior", "references", "contracts", "track", "record"
                ],
                phrases=[
                    "past performance", "volume iv", "volume 4", "performance history",
                    "contract history", "relevant experience", "similar work",
                    "performance references", "cpars", "past performance volume"
                ],
                volume_indicators=["volume iv", "volume 4", "vol iv", "vol 4", "volume four"],
                federal_conventions=[
                    "past performance questionnaire", "ppq", "cpars ratings",
                    "performance confidence assessment", "pca", "factor 4"
                ],
                confidence_weights={
                    "volume_indicator": 0.4,
                    "keywords": 0.3,
                    "phrases": 0.2,
                    "federal_convention": 0.1
                },
                format_indicators=["docx", "pdf", "doc"]
            ),

            DocumentType.COVER_LETTER: DocumentPattern(
                document_type=DocumentType.COVER_LETTER,
                keywords=[
                    "cover", "letter", "introduction", "transmittal", "executive",
                    "summary", "overview", "brief", "highlights"
                ],
                phrases=[
                    "cover letter", "transmittal letter", "executive summary",
                    "proposal summary", "introduction letter", "covering letter"
                ],
                volume_indicators=[],
                federal_conventions=[
                    "contracting officer", "contracting specialist", "proposal submission",
                    "solicitation response", "rfp response"
                ],
                confidence_weights={
                    "keywords": 0.4,
                    "phrases": 0.3,
                    "federal_convention": 0.2,
                    "format": 0.1
                },
                format_indicators=["docx", "pdf", "doc"]
            ),

            DocumentType.PRICE_NARRATIVE: DocumentPattern(
                document_type=DocumentType.PRICE_NARRATIVE,
                keywords=[
                    "price", "narrative", "justification", "explanation", "rationale",
                    "basis", "estimates", "methodology", "assumptions"
                ],
                phrases=[
                    "price narrative", "cost narrative", "pricing justification",
                    "cost justification", "basis of estimate", "pricing methodology"
                ],
                volume_indicators=[],
                federal_conventions=[
                    "basis of estimate", "boe", "cost estimating methodology",
                    "price reasonableness", "cost analysis"
                ],
                confidence_weights={
                    "phrases": 0.4,
                    "keywords": 0.3,
                    "federal_convention": 0.2,
                    "format": 0.1
                },
                format_indicators=["docx", "pdf", "doc"]
            ),

            DocumentType.CAPABILITY_STATEMENT: DocumentPattern(
                document_type=DocumentType.CAPABILITY_STATEMENT,
                keywords=[
                    "capability", "capabilities", "qualifications", "expertise",
                    "competencies", "skills", "resources", "facilities"
                ],
                phrases=[
                    "capability statement", "capabilities overview", "company capabilities",
                    "organizational capabilities", "technical capabilities"
                ],
                volume_indicators=[],
                federal_conventions=[
                    "core competencies", "differentiators", "competitive advantages",
                    "unique qualifications", "specialized expertise"
                ],
                confidence_weights={
                    "phrases": 0.4,
                    "keywords": 0.3,
                    "federal_convention": 0.2,
                    "format": 0.1
                },
                format_indicators=["docx", "pdf", "doc"]
            ),

            DocumentType.VIDEO_PITCH: DocumentPattern(
                document_type=DocumentType.VIDEO_PITCH,
                keywords=[
                    "video", "pitch", "presentation", "multimedia", "visual",
                    "demonstration", "overview", "summary"
                ],
                phrases=[
                    "video pitch", "video presentation", "multimedia presentation",
                    "video summary", "pitch video", "demonstration video"
                ],
                volume_indicators=[],
                federal_conventions=[
                    "capability demonstration", "solution overview",
                    "value proposition", "competitive advantages"
                ],
                confidence_weights={
                    "phrases": 0.4,
                    "keywords": 0.3,
                    "format": 0.2,
                    "federal_convention": 0.1
                },
                format_indicators=["mp4", "avi", "mov", "wmv", "script", "storyboard"]
            )
        }

    def _initialize_volume_patterns(self) -> Dict[str, str]:
        """Initialize volume structure patterns"""
        return {
            # Roman numerals
            r'\bvolume\s+i\b|\bvol\s+i\b|\bvolume\s+one\b': "technical",
            r'\bvolume\s+ii\b|\bvol\s+ii\b|\bvolume\s+two\b': "cost",
            r'\bvolume\s+iii\b|\bvol\s+iii\b|\bvolume\s+three\b': "management",
            r'\bvolume\s+iv\b|\bvol\s+iv\b|\bvolume\s+four\b': "past_performance",
            r'\bvolume\s+v\b|\bvol\s+v\b|\bvolume\s+five\b': "additional",

            # Numeric
            r'\bvolume\s+1\b|\bvol\s+1\b': "technical",
            r'\bvolume\s+2\b|\bvol\s+2\b': "cost",
            r'\bvolume\s+3\b|\bvol\s+3\b': "management",
            r'\bvolume\s+4\b|\bvol\s+4\b': "past_performance",
            r'\bvolume\s+5\b|\bvol\s+5\b': "additional",

            # Alpha
            r'\bvolume\s+a\b|\bvol\s+a\b': "technical",
            r'\bvolume\s+b\b|\bvol\s+b\b': "cost",
            r'\bvolume\s+c\b|\bvol\s+c\b': "management",
            r'\bvolume\s+d\b|\bvol\s+d\b': "past_performance"
        }

    def _initialize_federal_conventions(self) -> Dict[str, List[str]]:
        """Initialize federal contracting conventions"""
        return {
            "evaluation_factors": [
                "factor 1", "factor 2", "factor 3", "factor 4",
                "technical approach", "cost/price", "management approach",
                "past performance", "small business participation"
            ],
            "submission_types": [
                "technical proposal", "cost proposal", "price proposal",
                "management proposal", "past performance volume"
            ],
            "document_formats": [
                "cover letter", "executive summary", "white paper",
                "capability statement", "quad chart", "one-pager"
            ],
            "compliance_documents": [
                "sf-18", "sf-1449", "dd form 254", "representations and certifications",
                "wage determination", "security plan", "quality assurance plan"
            ]
        }


class EnhancedDocumentIdentifier:
    """Enhanced document identification using GPT-5 intelligence"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        self.registry = FederalDocumentRegistry()

    async def identify_required_documents(self, northstar: str, rfp_metadata: Dict[str, Any] = None) -> List[DocumentIdentification]:
        """
        Enhanced document identification using GPT-5 intelligent analysis
        Replaces primitive keyword matching with comprehensive pattern recognition
        """
        try:
            logger.info("Starting enhanced document identification with GPT-5 intelligence")

            # First, analyze volume structure
            volume_structure = await self._detect_volume_structure(northstar)

            # Run GPT-5 intelligent analysis
            gpt5_analysis = await self._run_gpt5_analysis(northstar, rfp_metadata, volume_structure)

            # Apply pattern-based validation
            pattern_validation = await self._validate_with_patterns(gpt5_analysis, northstar)

            # Combine and refine results
            final_documents = await self._refine_document_list(
                gpt5_analysis, pattern_validation, volume_structure, northstar
            )

            # Add confidence scores and validation
            scored_documents = await self._add_confidence_scoring(final_documents, northstar)

            logger.info(f"Enhanced identification complete: {len(scored_documents)} documents identified")
            return scored_documents

        except Exception as e:
            logger.error(f"Enhanced document identification failed: {e}")
            # Fallback to classical approach
            return await self._fallback_to_classical(northstar)

    async def _detect_volume_structure(self, northstar: str) -> VolumeStructure:
        """Detect volume structure in requirements"""
        try:
            text_lower = northstar.lower()
            volume_mapping = {}
            volume_count = 0
            numbering_style = "unknown"

            # Check for volume patterns
            for pattern, content_type in self.registry.volume_patterns.items():
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    volume_count += len(set(matches))
                    for match in set(matches):
                        volume_mapping[match] = content_type

                    # Determine numbering style
                    if re.search(r'\bi{1,4}\b', match):
                        numbering_style = "roman"
                    elif re.search(r'\b\d\b', match):
                        numbering_style = "numeric"
                    elif re.search(r'\b[a-d]\b', match):
                        numbering_style = "alpha"

            has_volumes = volume_count > 0

            logger.info(f"Volume structure detected: {has_volumes}, count: {volume_count}, style: {numbering_style}")

            return VolumeStructure(
                has_volumes=has_volumes,
                volume_count=volume_count,
                volume_mapping=volume_mapping,
                numbering_style=numbering_style
            )

        except Exception as e:
            logger.error(f"Error detecting volume structure: {e}")
            return VolumeStructure(False, 0, {}, "unknown")

    async def _run_gpt5_analysis(self, northstar: str, rfp_metadata: Dict[str, Any], volume_structure: VolumeStructure) -> Dict[str, Any]:
        """Run GPT-5 intelligent document analysis"""
        try:
            system_prompt = """You are an expert federal contracting document analyst with deep knowledge of:
            - Federal acquisition regulations (FAR)
            - Government solicitation patterns
            - Document submission requirements
            - Volume structure conventions
            - Evaluation criteria alignment

            Analyze solicitation requirements to identify required documents with high precision."""

            analysis_prompt = f"""
            Analyze this government solicitation to identify ALL required documents with intelligent reasoning.

            SOLICITATION ANALYSIS:
            {northstar[:15000]}  # Limit for context

            METADATA (if available):
            {json.dumps(rfp_metadata or {}, indent=2)}

            DETECTED VOLUME STRUCTURE:
            - Has Volumes: {volume_structure.has_volumes}
            - Volume Count: {volume_structure.volume_count}
            - Numbering Style: {volume_structure.numbering_style}
            - Volume Mapping: {volume_structure.volume_mapping}

            INTELLIGENT ANALYSIS FRAMEWORK:

            1. SOLICITATION TYPE ANALYSIS:
               - Determine: RFI, RFQ, RFP, Sources Sought, etc.
               - Identify submission expectations for this type
               - Note any special requirements or formats

            2. EXPLICIT DOCUMENT REQUIREMENTS:
               - Extract EXACT document names mentioned
               - Identify format specifications (PDF, DOCX, XLSX, etc.)
               - Note page limits, naming conventions
               - Find submission instructions

            3. VOLUME STRUCTURE INTELLIGENCE:
               - If volumes detected, map content to volumes
               - Understand volume purposes and boundaries
               - Identify volume-specific requirements
               - Consider evaluation factor alignment

            4. FEDERAL PATTERN RECOGNITION:
               - Recognize standard federal document types
               - Identify cover letters, narratives, statements
               - Detect compliance matrices, security plans
               - Recognize price vs. cost distinctions

            5. EVALUATION CRITERIA ALIGNMENT:
               - Map documents to evaluation factors
               - Understand what evaluators need to see
               - Consider evaluation team structure
               - Align document boundaries with review process

            6. AMBIGUITY RESOLUTION:
               - Identify unclear or contradictory requirements
               - Make reasonable assumptions based on federal conventions
               - Flag areas needing human clarification
               - Provide confidence levels for decisions

            Return JSON with this structure:
            {{
                "solicitation_analysis": {{
                    "type": "RFI/RFP/RFQ/etc",
                    "submission_method": "portal/email/mail",
                    "evaluation_approach": "best_value/lowest_price/etc",
                    "special_requirements": ["list of unique aspects"]
                }},
                "volume_analysis": {{
                    "uses_volumes": true/false,
                    "volume_structure": "roman/numeric/alpha/mixed",
                    "volume_contents": {{"volume_id": "content_description"}},
                    "evaluation_alignment": "how volumes align with factors"
                }},
                "document_requirements": [
                    {{
                        "name": "exact_filename_with_extension",
                        "type": "technical_volume/cost_volume/cover_letter/etc",
                        "requirements": "detailed content requirements",
                        "format": "pdf/docx/xlsx/etc",
                        "page_limit": null or number,
                        "evaluation_factor": "which factor this supports",
                        "confidence": 0.0-1.0,
                        "reasoning": "why this document is needed",
                        "volume_assignment": "if part of volume structure",
                        "mandatory": true/false,
                        "assumptions": ["list of assumptions made"]
                    }}
                ],
                "ambiguities": [
                    {{
                        "issue": "description of unclear requirement",
                        "assumption": "assumption made to resolve",
                        "confidence": 0.0-1.0,
                        "needs_clarification": true/false
                    }}
                ],
                "overall_confidence": 0.0-1.0,
                "human_review_needed": true/false
            }}

            BE EXTREMELY THOROUGH. Missing a required document could disqualify the proposal.
            Think step-by-step through the requirements and apply federal contracting expertise.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=16000,
                reasoning_effort="high" if "gpt-5" in self.model else None
            )

            analysis = json.loads(response.choices[0].message.content)
            logger.info(f"GPT-5 analysis complete. Confidence: {analysis.get('overall_confidence', 'N/A')}")

            return analysis

        except Exception as e:
            logger.error(f"GPT-5 analysis failed: {e}")
            raise

    async def _validate_with_patterns(self, gpt5_analysis: Dict[str, Any], northstar: str) -> Dict[str, Any]:
        """Validate GPT-5 results against known patterns"""
        try:
            validation_results = {
                "pattern_matches": {},
                "confidence_adjustments": {},
                "missing_patterns": [],
                "unexpected_documents": []
            }

            text_lower = northstar.lower()

            # Check each GPT-5 identified document against patterns
            for doc in gpt5_analysis.get("document_requirements", []):
                doc_name = doc.get("name", "").lower()
                doc_type = doc.get("type", "")

                best_pattern_match = None
                best_confidence = 0.0

                # Test against all patterns
                for pattern_type, pattern in self.registry.patterns.items():
                    confidence = self._calculate_pattern_confidence(
                        doc_name, doc.get("requirements", ""), pattern, text_lower
                    )

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_pattern_match = pattern_type

                validation_results["pattern_matches"][doc.get("name", "")] = {
                    "best_match": best_pattern_match.value if best_pattern_match else None,
                    "confidence": best_confidence,
                    "gpt5_type": doc_type
                }

                # Adjust confidence based on pattern match
                if best_confidence > 0.7:
                    validation_results["confidence_adjustments"][doc.get("name", "")] = min(1.0, doc.get("confidence", 0.5) + 0.2)
                elif best_confidence < 0.3:
                    validation_results["confidence_adjustments"][doc.get("name", "")] = max(0.1, doc.get("confidence", 0.5) - 0.2)

            # Check for missing common patterns
            for pattern_type, pattern in self.registry.patterns.items():
                found_in_gpt5 = any(
                    pattern_type.value in doc.get("type", "").lower()
                    for doc in gpt5_analysis.get("document_requirements", [])
                )

                if not found_in_gpt5:
                    pattern_confidence = self._calculate_pattern_confidence("", "", pattern, text_lower)
                    if pattern_confidence > 0.6:
                        validation_results["missing_patterns"].append({
                            "type": pattern_type.value,
                            "confidence": pattern_confidence,
                            "reason": "High pattern match but not identified by GPT-5"
                        })

            logger.info(f"Pattern validation complete. Missing patterns: {len(validation_results['missing_patterns'])}")
            return validation_results

        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            return {"pattern_matches": {}, "confidence_adjustments": {}, "missing_patterns": [], "unexpected_documents": []}

    def _calculate_pattern_confidence(self, doc_name: str, requirements: str, pattern: DocumentPattern, text: str) -> float:
        """Calculate confidence score for pattern match"""
        try:
            scores = {}
            total_weight = sum(pattern.confidence_weights.values())

            # Keyword matching
            keyword_score = 0
            for keyword in pattern.keywords:
                if keyword in doc_name or keyword in requirements or keyword in text:
                    keyword_score += 1
            keyword_score = min(1.0, keyword_score / max(len(pattern.keywords), 1))
            scores["keywords"] = keyword_score * pattern.confidence_weights.get("keywords", 0.3)

            # Phrase matching
            phrase_score = 0
            for phrase in pattern.phrases:
                if phrase in doc_name or phrase in requirements or phrase in text:
                    phrase_score += 1
            phrase_score = min(1.0, phrase_score / max(len(pattern.phrases), 1))
            scores["phrases"] = phrase_score * pattern.confidence_weights.get("phrases", 0.2)

            # Volume indicator matching
            volume_score = 0
            for volume_indicator in pattern.volume_indicators:
                if volume_indicator in text:
                    volume_score += 1
            volume_score = min(1.0, volume_score / max(len(pattern.volume_indicators), 1)) if pattern.volume_indicators else 0
            scores["volume_indicator"] = volume_score * pattern.confidence_weights.get("volume_indicator", 0.4)

            # Federal convention matching
            federal_score = 0
            for convention in pattern.federal_conventions:
                if convention in text:
                    federal_score += 1
            federal_score = min(1.0, federal_score / max(len(pattern.federal_conventions), 1))
            scores["federal_convention"] = federal_score * pattern.confidence_weights.get("federal_convention", 0.1)

            total_score = sum(scores.values()) / total_weight if total_weight > 0 else 0
            return total_score

        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.0

    async def _refine_document_list(self, gpt5_analysis: Dict[str, Any], pattern_validation: Dict[str, Any],
                                  volume_structure: VolumeStructure, northstar: str) -> List[Dict[str, Any]]:
        """Refine and consolidate document list"""
        try:
            refined_documents = []

            # Start with GPT-5 identified documents
            for doc in gpt5_analysis.get("document_requirements", []):
                doc_name = doc.get("name", "")

                # Apply confidence adjustments from pattern validation
                adjusted_confidence = pattern_validation.get("confidence_adjustments", {}).get(doc_name, doc.get("confidence", 0.5))

                refined_doc = {
                    "name": doc_name,
                    "type": doc.get("type", "unknown"),
                    "requirements": doc.get("requirements", ""),
                    "format": doc.get("format", "docx"),
                    "confidence": adjusted_confidence,
                    "reasoning": doc.get("reasoning", ""),
                    "page_limit": doc.get("page_limit"),
                    "specialized": self._needs_specialized_agent(doc),
                    "consolidates": doc.get("consolidates", []),
                    "assumptions": doc.get("assumptions", []),
                    "volume_assignment": doc.get("volume_assignment"),
                    "mandatory": doc.get("mandatory", True)
                }

                refined_documents.append(refined_doc)

            # Add high-confidence missing patterns
            for missing_pattern in pattern_validation.get("missing_patterns", []):
                if missing_pattern["confidence"] > 0.7:
                    pattern_type = missing_pattern["type"]
                    refined_documents.append({
                        "name": f"{pattern_type.replace('_', ' ').title()}.docx",
                        "type": pattern_type,
                        "requirements": f"Requirements for {pattern_type} based on pattern detection",
                        "format": "docx",
                        "confidence": missing_pattern["confidence"],
                        "reasoning": missing_pattern["reason"],
                        "page_limit": None,
                        "specialized": True,
                        "consolidates": [],
                        "assumptions": ["Added based on pattern matching"],
                        "volume_assignment": None,
                        "mandatory": False
                    })

            # Remove duplicates and low-confidence documents
            final_documents = []
            seen_types = set()

            # Sort by confidence (highest first)
            refined_documents.sort(key=lambda x: x["confidence"], reverse=True)

            for doc in refined_documents:
                doc_type = doc["type"]

                # Skip if we already have this type and current confidence is low
                if doc_type in seen_types and doc["confidence"] < 0.6:
                    continue

                # Skip if confidence is too low
                if doc["confidence"] < 0.3:
                    continue

                seen_types.add(doc_type)
                final_documents.append(doc)

            logger.info(f"Document list refined: {len(refined_documents)} -> {len(final_documents)} documents")
            return final_documents

        except Exception as e:
            logger.error(f"Error refining document list: {e}")
            return gpt5_analysis.get("document_requirements", [])

    async def _add_confidence_scoring(self, documents: List[Dict[str, Any]], northstar: str) -> List[DocumentIdentification]:
        """Add comprehensive confidence scoring and create final document identifications"""
        try:
            final_documents = []

            for doc in documents:
                # Create DocumentIdentification object
                document_id = DocumentIdentification(
                    name=doc["name"],
                    document_type=DocumentType(doc["type"]) if doc["type"] in [dt.value for dt in DocumentType] else DocumentType.TECHNICAL_PROPOSAL,
                    requirements=doc["requirements"],
                    format=doc["format"],
                    confidence_score=doc["confidence"],
                    reasoning=doc["reasoning"],
                    volume_info=None,  # Would be populated if part of volume structure
                    page_limit=doc.get("page_limit"),
                    specialized=doc["specialized"],
                    consolidates=doc["consolidates"],
                    assumptions=doc["assumptions"]
                )

                final_documents.append(document_id)

            logger.info(f"Confidence scoring complete for {len(final_documents)} documents")
            return final_documents

        except Exception as e:
            logger.error(f"Error adding confidence scoring: {e}")
            return []

    def _needs_specialized_agent(self, doc_spec: Dict[str, Any]) -> bool:
        """Determine if document needs specialized agent"""
        specialized_types = {
            "technical_volume", "cost_volume", "management_volume",
            "past_performance", "security_plan", "video_pitch",
            "price_narrative", "compliance_matrix"
        }

        return doc_spec.get("type", "") in specialized_types

    async def _fallback_to_classical(self, northstar: str) -> List[DocumentIdentification]:
        """Fallback to classical orchestrator approach"""
        try:
            logger.warning("Falling back to classical document identification")

            # Basic fallback documents
            fallback_docs = [
                DocumentIdentification(
                    name="Technical_Proposal.docx",
                    document_type=DocumentType.TECHNICAL_PROPOSAL,
                    requirements="Technical approach and solution",
                    format="docx",
                    confidence_score=0.6,
                    reasoning="Fallback to classical approach",
                    volume_info=None,
                    page_limit=None,
                    specialized=True,
                    consolidates=[],
                    assumptions=["Fallback identification"]
                ),
                DocumentIdentification(
                    name="Cost_Proposal.xlsx",
                    document_type=DocumentType.COST_VOLUME,
                    requirements="Detailed cost breakdown",
                    format="xlsx",
                    confidence_score=0.6,
                    reasoning="Fallback to classical approach",
                    volume_info=None,
                    page_limit=None,
                    specialized=True,
                    consolidates=[],
                    assumptions=["Fallback identification"]
                ),
                DocumentIdentification(
                    name="Executive_Summary.pdf",
                    document_type=DocumentType.EXECUTIVE_SUMMARY,
                    requirements="High-level overview",
                    format="pdf",
                    confidence_score=0.6,
                    reasoning="Fallback to classical approach",
                    volume_info=None,
                    page_limit=None,
                    specialized=False,
                    consolidates=[],
                    assumptions=["Fallback identification"]
                )
            ]

            return fallback_docs

        except Exception as e:
            logger.error(f"Fallback identification failed: {e}")
            return []
