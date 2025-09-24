"""
Comprehensive Document Review Agent for holistic evaluation and human feedback integration
"""

import os
import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class DocumentGrade(BaseModel):
    """Represents a multi-dimensional grade for a document"""
    accuracy: float = Field(description="Score for factual accuracy (0-100)")
    completeness: float = Field(description="Score for requirement coverage (0-100)")
    specificity: float = Field(description="Score for concrete details vs vague claims (0-100)")
    compliance: float = Field(description="Score for RFP compliance (0-100)")
    coherence: float = Field(description="Score for narrative consistency (0-100)")
    northstar_alignment: float = Field(description="Score for Northstar specification alignment (0-100)")
    overall_score: float = Field(description="Weighted overall score (0-100)")
    grade_tier: str = Field(description="Quality tier: excellent|good|acceptable|below_standard|poor")

class WeaknessPattern(BaseModel):
    """Represents a detected weakness in a document"""
    type: str = Field(description="Type of weakness: vague_claim|missing_info|potential_hallucination|compliance_gap")
    severity: str = Field(description="Severity level: critical|high|medium|low")
    location: str = Field(description="Document and section where weakness found")
    description: str = Field(description="Description of the weakness")
    suggested_question: str = Field(description="Question to ask human for improvement")
    confidence: float = Field(description="Confidence in weakness detection (0-1)")

class CrossDocumentIssue(BaseModel):
    """Represents an issue found across multiple documents"""
    type: str = Field(description="Type of issue: contradiction|redundancy|gap|inconsistency")
    documents_affected: List[str] = Field(description="List of affected documents")
    description: str = Field(description="Description of the cross-document issue")
    severity: str = Field(description="Severity level")
    resolution_needed: str = Field(description="What needs to be resolved")

class HumanQuestion(BaseModel):
    """Represents a targeted question for human input"""
    id: str = Field(description="Unique question ID")
    category: str = Field(description="Category: accuracy|completeness|specificity|compliance|clarification")
    priority: str = Field(description="Priority: critical|high|medium|low")
    question: str = Field(description="The actual question to ask")
    context: str = Field(description="Context for the question")
    document_section: Optional[str] = Field(description="Specific document section if applicable")
    improvement_potential: float = Field(description="Potential grade improvement if answered (0-100)")

class ReviewReport(BaseModel):
    """Comprehensive review report for document suite"""
    timestamp: datetime
    individual_grades: Dict[str, DocumentGrade]
    cross_document_issues: List[CrossDocumentIssue]
    weakness_patterns: List[WeaknessPattern]
    human_questions: List[HumanQuestion]
    overall_suite_grade: float
    compliance_status: bool
    revision_recommendations: List[str]


class ComprehensiveReviewAgent:
    """Agent for comprehensive document review and human feedback integration"""

    def __init__(self, milvus_rag=None):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=float(os.getenv("OPENAI_TIMEOUT", "180")))
        self.milvus_rag = milvus_rag
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Grading weights for overall score
        self.grading_weights = {
            'accuracy': 0.25,
            'completeness': 0.20,
            'specificity': 0.15,
            'compliance': 0.25,
            'coherence': 0.10,
            'northstar_alignment': 0.05
        }

        # Patterns for weakness detection
        self.vague_patterns = [
            r'\b(extensive|significant|comprehensive|robust|proven|industry-leading)\s+\w+',
            r'\b(various|multiple|several|numerous|many)\s+\w+',
            r'\b(typically|generally|usually|often|commonly)\b',
            r'\b(approximately|around|roughly|about)\s+\d+',
        ]

        self.hallucination_indicators = [
            "might", "could", "should", "would", "may", "possibly", "potentially",
            "estimated", "projected", "anticipated", "expected"
        ]

    async def review_document_suite(
        self,
        documents: Dict[str, str],
        northstar_doc: str,
        rfp_metadata: Dict[str, Any]
    ) -> ReviewReport:
        """
        Perform comprehensive review of all documents
        """
        logger.info(f"Starting comprehensive review of {len(documents)} documents")

        # Step 1: Individual document analysis
        individual_grades = {}
        for doc_name, content in documents.items():
            grade = await self._evaluate_single_document(
                doc_name, content, northstar_doc, rfp_metadata
            )
            individual_grades[doc_name] = grade
            logger.info(f"Graded {doc_name}: Overall score {grade.overall_score:.1f}")

        # Step 2: Cross-document analysis
        cross_document_issues = await self._analyze_cross_document_consistency(
            documents, northstar_doc
        )

        # Step 3: Weakness pattern detection
        weakness_patterns = await self._detect_weakness_patterns(
            documents, individual_grades
        )

        # Step 4: Generate human questions
        human_questions = await self._generate_human_questions(
            weakness_patterns,
            cross_document_issues,
            individual_grades,
            rfp_metadata
        )

        # Step 5: Calculate overall suite grade
        overall_suite_grade = self._calculate_suite_grade(individual_grades)

        # Step 6: Determine compliance status
        compliance_status = all(
            grade.compliance >= 85 for grade in individual_grades.values()
        )

        # Step 7: Generate revision recommendations
        revision_recommendations = self._generate_revision_recommendations(
            individual_grades, cross_document_issues, weakness_patterns
        )

        return ReviewReport(
            timestamp=datetime.now(),
            individual_grades=individual_grades,
            cross_document_issues=cross_document_issues,
            weakness_patterns=weakness_patterns,
            human_questions=human_questions,
            overall_suite_grade=overall_suite_grade,
            compliance_status=compliance_status,
            revision_recommendations=revision_recommendations
        )

    async def _evaluate_single_document(
        self,
        doc_name: str,
        content: str,
        northstar_doc: str,
        rfp_metadata: Dict[str, Any]
    ) -> DocumentGrade:
        """Evaluate a single document on multiple dimensions"""

        # Truncate content for API limits
        content_sample = content[:8000] if len(content) > 8000 else content
        northstar_sample = northstar_doc[:3000] if len(northstar_doc) > 3000 else northstar_doc

        prompt = f"""
        Evaluate this RFP response document on multiple quality dimensions.

        DOCUMENT: {doc_name}
        CONTENT: {content_sample}

        NORTHSTAR REQUIREMENTS: {northstar_sample}

        RFP CONTEXT: {json.dumps(rfp_metadata, indent=2)[:1000]}

        Evaluate on these dimensions (0-100 scale):

        1. ACCURACY: Are all claims factual and verifiable? Look for:
           - Unsubstantiated claims
           - Exaggerations
           - Technical inaccuracies

        2. COMPLETENESS: Does it address all requirements? Look for:
           - Missing sections
           - Unanswered requirements
           - Incomplete information

        3. SPECIFICITY: Are claims concrete with evidence? Look for:
           - Vague statements vs specific examples
           - Quantified metrics vs generalizations
           - Named tools/methods vs generic descriptions

        4. COMPLIANCE: Does it meet RFP specifications? Look for:
           - Format requirements
           - Mandatory sections
           - Submission guidelines

        5. COHERENCE: Is the narrative consistent? Look for:
           - Logical flow
           - Consistent messaging
           - Professional tone

        6. NORTHSTAR_ALIGNMENT: How well does it follow the Northstar document guidance?

        Return JSON with scores and specific issues found:
        {{
            "accuracy": <0-100>,
            "completeness": <0-100>,
            "specificity": <0-100>,
            "compliance": <0-100>,
            "coherence": <0-100>,
            "northstar_alignment": <0-100>,
            "critical_issues": ["list of critical problems"],
            "improvement_areas": ["list of areas needing improvement"],
            "strengths": ["list of strong points"]
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)

            # Calculate weighted overall score
            overall_score = sum(
                result.get(dim, 0) * self.grading_weights[dim]
                for dim in self.grading_weights
            )

            # Determine grade tier
            if overall_score >= 90:
                grade_tier = "excellent"
            elif overall_score >= 80:
                grade_tier = "good"
            elif overall_score >= 70:
                grade_tier = "acceptable"
            elif overall_score >= 60:
                grade_tier = "below_standard"
            else:
                grade_tier = "poor"

            return DocumentGrade(
                accuracy=result.get('accuracy', 0),
                completeness=result.get('completeness', 0),
                specificity=result.get('specificity', 0),
                compliance=result.get('compliance', 0),
                coherence=result.get('coherence', 0),
                northstar_alignment=result.get('northstar_alignment', 0),
                overall_score=overall_score,
                grade_tier=grade_tier
            )

        except Exception as e:
            logger.error(f"Error evaluating {doc_name}: {e}")
            # Return a conservative grade on error
            return DocumentGrade(
                accuracy=60, completeness=60, specificity=60,
                compliance=60, coherence=60, northstar_alignment=60,
                overall_score=60, grade_tier="below_standard"
            )

    async def _analyze_cross_document_consistency(
        self,
        documents: Dict[str, str],
        northstar_doc: str
    ) -> List[CrossDocumentIssue]:
        """Analyze consistency across all documents"""

        issues = []

        # Prepare document summaries for cross-reference
        doc_summaries = {}
        for doc_name, content in documents.items():
            # Extract key claims and data points
            doc_summaries[doc_name] = content[:2000]  # Use first 2000 chars as summary

        prompt = f"""
        Analyze these RFP response documents for cross-document consistency issues.

        DOCUMENTS:
        {json.dumps(doc_summaries, indent=2)}

        NORTHSTAR GUIDANCE: {northstar_doc[:2000]}

        Look for:
        1. CONTRADICTIONS: Conflicting information between documents
        2. REDUNDANCIES: Unnecessary repetition across documents
        3. GAPS: Missing connections or references between related documents
        4. INCONSISTENCIES: Different formatting, tone, or terminology

        Return JSON array of issues found:
        {{
            "issues": [
                {{
                    "type": "contradiction|redundancy|gap|inconsistency",
                    "documents_affected": ["doc1", "doc2"],
                    "description": "Specific description of the issue",
                    "severity": "critical|high|medium|low",
                    "resolution_needed": "What needs to be fixed"
                }}
            ]
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000
            )

            result = json.loads(response.choices[0].message.content)

            for issue_data in result.get('issues', []):
                issues.append(CrossDocumentIssue(**issue_data))

        except Exception as e:
            logger.error(f"Error in cross-document analysis: {e}")

        return issues

    async def _detect_weakness_patterns(
        self,
        documents: Dict[str, str],
        grades: Dict[str, DocumentGrade]
    ) -> List[WeaknessPattern]:
        """Detect specific weakness patterns in documents"""

        weaknesses = []

        for doc_name, content in documents.items():
            grade = grades[doc_name]

            # Check for vague claims
            for pattern in self.vague_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    context = content[max(0, match.start()-50):min(len(content), match.end()+50)]
                    weaknesses.append(WeaknessPattern(
                        type="vague_claim",
                        severity="medium" if grade.specificity < 70 else "low",
                        location=f"{doc_name} (char {match.start()})",
                        description=f"Vague claim: '{match.group()}'",
                        suggested_question=f"Can you provide specific metrics or examples for: {match.group()}?",
                        confidence=0.8
                    ))

            # Check for potential hallucinations
            for indicator in self.hallucination_indicators:
                if indicator in content.lower():
                    count = content.lower().count(indicator)
                    if count > 3:  # Multiple uncertain statements
                        weaknesses.append(WeaknessPattern(
                            type="potential_hallucination",
                            severity="high" if grade.accuracy < 70 else "medium",
                            location=doc_name,
                            description=f"Document contains {count} instances of uncertain language ('{indicator}')",
                            suggested_question=f"Please verify and provide concrete data for claims using '{indicator}'",
                            confidence=0.7
                        ))

            # Check for missing information based on low grades
            if grade.completeness < 70:
                weaknesses.append(WeaknessPattern(
                    type="missing_info",
                    severity="high",
                    location=doc_name,
                    description=f"Document completeness score is low ({grade.completeness:.0f}/100)",
                    suggested_question=f"What additional information can you provide to make {doc_name} more complete?",
                    confidence=0.9
                ))

            # Check for compliance gaps
            if grade.compliance < 85:
                weaknesses.append(WeaknessPattern(
                    type="compliance_gap",
                    severity="critical",
                    location=doc_name,
                    description=f"Document may not meet all RFP compliance requirements ({grade.compliance:.0f}/100)",
                    suggested_question=f"Please review {doc_name} for RFP compliance. Are all mandatory requirements addressed?",
                    confidence=0.95
                ))

        return weaknesses

    async def _generate_human_questions(
        self,
        weakness_patterns: List[WeaknessPattern],
        cross_document_issues: List[CrossDocumentIssue],
        grades: Dict[str, DocumentGrade],
        rfp_metadata: Dict[str, Any]
    ) -> List[HumanQuestion]:
        """Generate targeted questions for human input"""

        questions = []
        question_id = 1

        # Generate questions from weakness patterns
        for weakness in weakness_patterns:
            if weakness.severity in ["critical", "high"]:
                questions.append(HumanQuestion(
                    id=f"Q{question_id:03d}",
                    category=weakness.type.replace("_", " ").title(),
                    priority=weakness.severity,
                    question=weakness.suggested_question,
                    context=weakness.description,
                    document_section=weakness.location,
                    improvement_potential=self._estimate_improvement_potential(weakness)
                ))
                question_id += 1

        # Generate questions from cross-document issues
        for issue in cross_document_issues:
            if issue.severity in ["critical", "high"]:
                questions.append(HumanQuestion(
                    id=f"Q{question_id:03d}",
                    category="consistency",
                    priority=issue.severity,
                    question=f"How should we resolve this inconsistency: {issue.description}?",
                    context=f"Affects: {', '.join(issue.documents_affected)}",
                    document_section=None,
                    improvement_potential=15.0
                ))
                question_id += 1

        # Generate questions for low-scoring dimensions
        for doc_name, grade in grades.items():
            if grade.accuracy < 75:
                questions.append(HumanQuestion(
                    id=f"Q{question_id:03d}",
                    category="accuracy",
                    priority="high",
                    question=f"Please verify the accuracy of claims in {doc_name}. Can you provide evidence for key assertions?",
                    context=f"Accuracy score: {grade.accuracy:.0f}/100",
                    document_section=doc_name,
                    improvement_potential=100 - grade.accuracy
                ))
                question_id += 1

            if grade.specificity < 70:
                questions.append(HumanQuestion(
                    id=f"Q{question_id:03d}",
                    category="specificity",
                    priority="medium",
                    question=f"Can you provide specific examples, metrics, or tools for {doc_name}?",
                    context=f"Specificity score: {grade.specificity:.0f}/100",
                    document_section=doc_name,
                    improvement_potential=100 - grade.specificity
                ))
                question_id += 1

        # Sort questions by priority and improvement potential
        questions.sort(key=lambda q: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}[q.priority],
            -q.improvement_potential
        ))

        return questions[:20]  # Return top 20 questions

    def _calculate_suite_grade(self, individual_grades: Dict[str, DocumentGrade]) -> float:
        """Calculate overall grade for document suite"""
        if not individual_grades:
            return 0.0

        # Average of all document scores with bonus for consistency
        avg_score = sum(g.overall_score for g in individual_grades.values()) / len(individual_grades)

        # Add bonus if all documents are above threshold
        if all(g.overall_score >= 70 for g in individual_grades.values()):
            avg_score = min(100, avg_score + 5)

        return avg_score

    def _generate_revision_recommendations(
        self,
        grades: Dict[str, DocumentGrade],
        cross_issues: List[CrossDocumentIssue],
        weaknesses: List[WeaknessPattern]
    ) -> List[str]:
        """Generate specific revision recommendations"""

        recommendations = []

        # Recommendations based on grades
        for doc_name, grade in grades.items():
            if grade.overall_score < 70:
                recommendations.append(
                    f"Major revision needed for {doc_name} (score: {grade.overall_score:.0f}/100)"
                )
            elif grade.overall_score < 80:
                recommendations.append(
                    f"Moderate improvements needed for {doc_name}"
                )

        # Recommendations for critical weaknesses
        critical_weaknesses = [w for w in weaknesses if w.severity == "critical"]
        if critical_weaknesses:
            recommendations.append(
                f"Address {len(critical_weaknesses)} critical issues immediately"
            )

        # Recommendations for cross-document issues
        critical_cross_issues = [i for i in cross_issues if i.severity == "critical"]
        if critical_cross_issues:
            recommendations.append(
                f"Resolve {len(critical_cross_issues)} critical cross-document inconsistencies"
            )

        return recommendations

    def _estimate_improvement_potential(self, weakness: WeaknessPattern) -> float:
        """Estimate potential grade improvement from addressing weakness"""

        severity_impact = {
            "critical": 20.0,
            "high": 15.0,
            "medium": 10.0,
            "low": 5.0
        }

        type_multiplier = {
            "compliance_gap": 1.5,
            "missing_info": 1.3,
            "potential_hallucination": 1.2,
            "vague_claim": 1.0
        }

        base_impact = severity_impact.get(weakness.severity, 5.0)
        multiplier = type_multiplier.get(weakness.type, 1.0)

        return min(30.0, base_impact * multiplier * weakness.confidence)

    async def process_human_feedback(
        self,
        original_documents: Dict[str, str],
        review_report: ReviewReport,
        human_responses: Dict[str, str]
    ) -> Tuple[Dict[str, str], ReviewReport]:
        """
        Process human feedback and update documents
        Returns updated documents and new review report
        """

        updated_documents = original_documents.copy()

        for question_id, response in human_responses.items():
            # Find the corresponding question
            question = next((q for q in review_report.human_questions if q.id == question_id), None)
            if not question:
                continue

            # Apply the feedback to relevant document
            if question.document_section:
                doc_name = question.document_section.split(" ")[0]  # Extract document name
                if doc_name in updated_documents:
                    updated_documents[doc_name] = await self._apply_feedback_to_document(
                        updated_documents[doc_name],
                        question,
                        response
                    )

        # Re-review the updated documents
        new_review_report = await self.review_document_suite(
            updated_documents,
            "",  # Would need to pass northstar_doc
            {}   # Would need to pass rfp_metadata
        )

        return updated_documents, new_review_report

    async def _apply_feedback_to_document(
        self,
        document_content: str,
        question: HumanQuestion,
        human_response: str
    ) -> str:
        """Apply human feedback to improve document content"""

        prompt = f"""
        Improve this document section based on human feedback.

        CURRENT CONTENT: {document_content[:4000]}

        QUESTION ASKED: {question.question}
        CONTEXT: {question.context}
        HUMAN RESPONSE: {human_response}

        Instructions:
        1. Incorporate the human's specific information naturally
        2. Replace vague statements with the concrete details provided
        3. Maintain the document's structure and flow
        4. Ensure professional tone and formatting
        5. Keep all existing good content

        Return the improved document content.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error applying feedback: {e}")
            return document_content  # Return original on error
