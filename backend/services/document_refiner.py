"""
Document Refiner for applying human feedback to improve documents
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

class RefinementResult(BaseModel):
    """Result of document refinement"""
    original_content: str
    refined_content: str
    changes_made: List[str]
    quality_improvement: float
    sections_modified: List[str]

class DocumentRefiner:
    """Service for refining documents based on human feedback"""

    def __init__(self, milvus_rag=None):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=float(os.getenv("OPENAI_TIMEOUT", "180")))
        self.milvus_rag = milvus_rag
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    async def refine_documents_with_feedback(
        self,
        session_id: str,
        documents: Dict[str, str],
        answered_questions: List[Dict[str, Any]],
        weaknesses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine documents based on human feedback
        """
        refined_documents = documents.copy()
        refinement_log = []

        # Group answers by document
        document_feedback = self._group_feedback_by_document(
            answered_questions, documents, weaknesses
        )

        # Refine each document that has feedback
        for doc_name, feedback_items in document_feedback.items():
            if doc_name in documents:
                original_content = documents[doc_name]

                # Apply refinements
                refined_content = await self._refine_document(
                    original_content,
                    feedback_items,
                    doc_name
                )

                refined_documents[doc_name] = refined_content

                # Log the refinement
                refinement_log.append({
                    'document': doc_name,
                    'feedback_count': len(feedback_items),
                    'timestamp': datetime.now().isoformat(),
                    'improvements': [f['improvement_type'] for f in feedback_items]
                })

        # Calculate quality improvement
        quality_metrics = await self._assess_quality_improvement(
            documents, refined_documents
        )

        return {
            'refined_documents': refined_documents,
            'refinement_log': refinement_log,
            'quality_metrics': quality_metrics,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }

    def _group_feedback_by_document(
        self,
        answered_questions: List[Dict[str, Any]],
        documents: Dict[str, str],
        weaknesses: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group feedback by relevant document"""

        document_feedback = {}

        for qa in answered_questions:
            question = qa.get('question', '')
            answer = qa.get('answer', '')

            # Determine which document this feedback applies to
            relevant_doc = self._find_relevant_document(
                question, documents, weaknesses
            )

            if relevant_doc:
                if relevant_doc not in document_feedback:
                    document_feedback[relevant_doc] = []

                document_feedback[relevant_doc].append({
                    'question': question,
                    'answer': answer,
                    'improvement_type': self._classify_improvement(question)
                })

        return document_feedback

    def _find_relevant_document(
        self,
        question: str,
        documents: Dict[str, str],
        weaknesses: Dict[str, Any]
    ) -> Optional[str]:
        """Find which document a question relates to"""

        question_lower = question.lower()

        # Check for document name mentions
        for doc_name in documents.keys():
            doc_name_clean = doc_name.replace('_', ' ').replace('.', ' ').lower()
            if doc_name_clean in question_lower:
                return doc_name

        # Check for content keywords
        keyword_mapping = {
            'technical': ['Technical_Proposal', 'Technical_Approach'],
            'cost': ['Cost_Proposal', 'Pricing'],
            'executive': ['Executive_Summary'],
            'compliance': ['Compliance_Matrix'],
            'management': ['Management_Plan'],
            'personnel': ['Key_Personnel', 'Staffing_Plan']
        }

        for keyword, doc_patterns in keyword_mapping.items():
            if keyword in question_lower:
                for doc_name in documents.keys():
                    for pattern in doc_patterns:
                        if pattern.lower() in doc_name.lower():
                            return doc_name

        # Default to first document if unclear
        return list(documents.keys())[0] if documents else None

    def _classify_improvement(self, question: str) -> str:
        """Classify the type of improvement based on question"""

        question_lower = question.lower()

        if any(word in question_lower for word in ['specific', 'detail', 'example']):
            return 'add_specifics'
        elif any(word in question_lower for word in ['verify', 'accurate', 'correct']):
            return 'verify_accuracy'
        elif any(word in question_lower for word in ['missing', 'add', 'include']):
            return 'add_missing_info'
        elif any(word in question_lower for word in ['clarify', 'explain', 'elaborate']):
            return 'clarify'
        elif any(word in question_lower for word in ['evidence', 'proof', 'support']):
            return 'add_evidence'
        else:
            return 'general_improvement'

    async def _refine_document(
        self,
        original_content: str,
        feedback_items: List[Dict[str, Any]],
        doc_name: str
    ) -> str:
        """Apply feedback to refine a document"""

        # Build refinement instructions
        refinement_instructions = self._build_refinement_instructions(feedback_items)

        prompt = f"""
        Refine this document based on human feedback to make it more compelling and specific.

        DOCUMENT: {doc_name}
        ORIGINAL CONTENT:
        {original_content[:6000]}

        HUMAN FEEDBACK TO INCORPORATE:
        {refinement_instructions}

        INSTRUCTIONS:
        1. Incorporate all the specific information provided by the human
        2. Replace vague statements with concrete details
        3. Add metrics, examples, and evidence where provided
        4. Maintain professional federal proposal language
        5. Ensure all claims are now substantiated
        6. Keep the original structure and all good content
        7. Make the document more compelling and win-focused

        Return the refined document content that incorporates all feedback.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=24000  # Quadrupled to prevent empty responses
            )

            refined_content = response.choices[0].message.content

            # Ensure we didn't lose content
            if len(refined_content) < len(original_content) * 0.7:
                logger.warning(f"Refined content significantly shorter for {doc_name}, using selective refinement")
                return await self._selective_refinement(
                    original_content, feedback_items, doc_name
                )

            return refined_content

        except Exception as e:
            logger.error(f"Error refining document {doc_name}: {e}")
            return original_content

    def _build_refinement_instructions(
        self,
        feedback_items: List[Dict[str, Any]]
    ) -> str:
        """Build clear instructions from feedback"""

        instructions = []

        for i, item in enumerate(feedback_items, 1):
            question = item['question']
            answer = item['answer']
            improvement_type = item['improvement_type']

            if improvement_type == 'add_specifics':
                instructions.append(
                    f"{i}. Replace vague claims with: {answer}"
                )
            elif improvement_type == 'verify_accuracy':
                instructions.append(
                    f"{i}. Update with verified information: {answer}"
                )
            elif improvement_type == 'add_missing_info':
                instructions.append(
                    f"{i}. Add this missing information: {answer}"
                )
            elif improvement_type == 'clarify':
                instructions.append(
                    f"{i}. Clarify with: {answer}"
                )
            elif improvement_type == 'add_evidence':
                instructions.append(
                    f"{i}. Support claims with this evidence: {answer}"
                )
            else:
                instructions.append(
                    f"{i}. Incorporate: {answer}"
                )

        return "\n".join(instructions)

    async def _selective_refinement(
        self,
        original_content: str,
        feedback_items: List[Dict[str, Any]],
        doc_name: str
    ) -> str:
        """Selectively refine specific sections"""

        refined_content = original_content

        for item in feedback_items:
            # Find the section that needs refinement
            section_to_refine = self._find_section_to_refine(
                original_content, item['question']
            )

            if section_to_refine:
                # Refine just this section
                refined_section = await self._refine_section(
                    section_to_refine,
                    item['question'],
                    item['answer']
                )

                # Replace in document
                refined_content = refined_content.replace(
                    section_to_refine,
                    refined_section
                )

        return refined_content

    def _find_section_to_refine(
        self,
        content: str,
        question: str
    ) -> Optional[str]:
        """Find the specific section that needs refinement"""

        # Look for keywords from question in content
        keywords = re.findall(r'\b\w{4,}\b', question.lower())

        paragraphs = content.split('\n\n')
        best_match = None
        best_score = 0

        for para in paragraphs:
            if len(para) < 50:
                continue

            para_lower = para.lower()
            score = sum(1 for keyword in keywords if keyword in para_lower)

            if score > best_score:
                best_score = score
                best_match = para

        return best_match if best_score > 2 else None

    async def _refine_section(
        self,
        section: str,
        question: str,
        answer: str
    ) -> str:
        """Refine a specific section"""

        prompt = f"""
        Improve this section based on human feedback.

        CURRENT SECTION:
        {section}

        QUESTION: {question}
        HUMAN ANSWER: {answer}

        Rewrite this section to incorporate the human's information while maintaining flow.
        Make it specific, compelling, and evidence-based.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000  # Quadrupled to prevent empty responses
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error refining section: {e}")
            return section

    async def _assess_quality_improvement(
        self,
        original_documents: Dict[str, str],
        refined_documents: Dict[str, str]
    ) -> Dict[str, Any]:
        """Assess the quality improvement from refinement"""

        improvements = {}

        for doc_name in original_documents.keys():
            if doc_name in refined_documents:
                original = original_documents[doc_name]
                refined = refined_documents[doc_name]

                # Count specific improvements
                improvements[doc_name] = {
                    'vague_statements_removed': self._count_vague_statements(original) - self._count_vague_statements(refined),
                    'specifics_added': self._count_specific_details(refined) - self._count_specific_details(original),
                    'length_change': len(refined) - len(original),
                    'metric_count': self._count_metrics(refined) - self._count_metrics(original)
                }

        # Calculate overall improvement score
        total_improvements = sum(
            max(0, imp['vague_statements_removed']) +
            max(0, imp['specifics_added']) +
            max(0, imp['metric_count'])
            for imp in improvements.values()
        )

        return {
            'document_improvements': improvements,
            'total_improvement_score': min(100, total_improvements * 5),
            'documents_refined': len([i for i in improvements.values() if any(v > 0 for v in i.values())])
        }

    def _count_vague_statements(self, content: str) -> int:
        """Count vague statements in content"""

        vague_patterns = [
            r'extensive experience',
            r'proven track record',
            r'comprehensive solution',
            r'various \w+',
            r'multiple \w+',
            r'several \w+'
        ]

        count = 0
        for pattern in vague_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _count_specific_details(self, content: str) -> int:
        """Count specific details in content"""

        # Count specific items like numbers, tool names, certifications
        specific_patterns = [
            r'\d+\s*(?:years?|months?|days?)',  # Time periods
            r'\d+%',  # Percentages
            r'\$[\d,]+',  # Dollar amounts
            r'[A-Z]{2,}(?:\s+\d+)?',  # Acronyms/certifications
            r'(?:AWS|Azure|GCP|Docker|Kubernetes)',  # Specific tools
        ]

        count = 0
        for pattern in specific_patterns:
            count += len(re.findall(pattern, content))

        return count

    def _count_metrics(self, content: str) -> int:
        """Count quantitative metrics in content"""

        # Count numbers that appear to be metrics
        metric_patterns = [
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\d+(?:x|X)',  # Multipliers
            r'(?:increased?|decreased?|improved?|reduced?)\s+(?:by\s+)?\d+',
        ]

        count = 0
        for pattern in metric_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    async def apply_single_refinement(
        self,
        document: str,
        question: str,
        answer: str
    ) -> str:
        """Apply a single refinement to a document"""

        prompt = f"""
        Improve this document by incorporating specific human feedback.

        DOCUMENT:
        {document[:4000]}

        HUMAN WAS ASKED: {question}
        HUMAN PROVIDED: {answer}

        Incorporate this information naturally into the document.
        Replace any vague statements with the specific details provided.
        Maintain document structure and professional tone.

        Return the improved document.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=16000  # Quadrupled to prevent empty responses
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in single refinement: {e}")
            return document
