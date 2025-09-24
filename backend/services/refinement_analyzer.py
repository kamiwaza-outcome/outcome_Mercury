"""
Refinement Analyzer for detecting weaknesses and generating targeted questions
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
import hashlib

logger = logging.getLogger(__name__)

class RefinementSession(BaseModel):
    """Represents a refinement session with human"""
    session_id: str
    notice_id: str
    created_at: datetime
    original_documents: Dict[str, str]
    review_report: Dict[str, Any]
    questions: List[Dict[str, Any]]
    answered_questions: List[Dict[str, Any]]
    refinements: List[Dict[str, Any]]
    status: str = "active"  # active, paused, completed

class ConversationEntry(BaseModel):
    """Represents a single conversation exchange"""
    timestamp: datetime
    question_id: str
    question: str
    human_response: str
    confidence_before: float
    confidence_after: float
    document_affected: str
    improvement_type: str

class KnowledgeExtraction(BaseModel):
    """Extracted knowledge from human feedback"""
    fact_type: str  # technical_spec, pricing_info, compliance_detail, team_info
    fact_content: str
    source: str  # human_verified, human_provided, human_corrected
    confidence: float
    applicable_contexts: List[str]
    save_to_rag: bool


class RefinementAnalyzer:
    """Analyzes documents for weaknesses and manages refinement sessions"""

    def __init__(self, milvus_rag=None):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=float(os.getenv("OPENAI_TIMEOUT", "180")))
        self.milvus_rag = milvus_rag
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.sessions = {}  # In-memory session storage

        # Confidence thresholds
        self.confidence_threshold = 0.7
        self.hallucination_threshold = 0.6

        # Weakness patterns
        self.weakness_indicators = {
            'vague_claims': [
                r'extensive experience',
                r'proven track record',
                r'industry.?leading',
                r'comprehensive solution',
                r'robust capabilities',
                r'significant expertise'
            ],
            'uncertain_language': [
                r'\b(approximately|around|roughly|about)\s+\d+',
                r'\b(typically|generally|usually|often|commonly)\b',
                r'\b(might|could|should|would|may|possibly|potentially)\b'
            ],
            'missing_specifics': [
                r'various\s+(tools|technologies|methods)',
                r'multiple\s+(clients|projects|solutions)',
                r'several\s+(years|projects|implementations)',
                r'numerous\s+(awards|certifications|contracts)'
            ]
        }

    async def analyze_for_refinement(
        self,
        notice_id: str,
        documents: Dict[str, str],
        northstar_doc: str,
        rfp_metadata: Dict[str, Any]
    ) -> RefinementSession:
        """
        Analyze documents and create refinement session
        """
        session_id = self._generate_session_id(notice_id)

        # Analyze weaknesses
        weaknesses = await self.analyze_weakness_patterns(
            documents, northstar_doc, rfp_metadata
        )

        # Generate targeted questions
        questions = await self.generate_targeted_questions(
            weaknesses, rfp_metadata
        )

        # Create review report
        review_report = {
            'weaknesses': weaknesses,
            'total_issues': len(weaknesses['all_issues']),
            'critical_count': len(weaknesses['critical_issues']),
            'improvement_potential': weaknesses['improvement_potential']
        }

        # Create session
        session = RefinementSession(
            session_id=session_id,
            notice_id=notice_id,
            created_at=datetime.now(),
            original_documents=documents,
            review_report=review_report,
            questions=questions,
            answered_questions=[],
            refinements=[],
            status="active"
        )

        self.sessions[session_id] = session
        return session

    async def analyze_weakness_patterns(
        self,
        documents: Dict[str, str],
        northstar_doc: str,
        rfp_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive weakness analysis across all documents
        """
        weaknesses = {
            'missing_information': [],
            'low_confidence_sections': [],
            'potential_hallucinations': [],
            'unsubstantiated_claims': [],
            'unaddressed_requirements': [],
            'vague_statements': [],
            'all_issues': [],
            'critical_issues': [],
            'improvement_potential': 0
        }

        for doc_name, content in documents.items():
            # Detect missing information
            missing = await self._detect_missing_information(
                content, northstar_doc, rfp_metadata
            )
            weaknesses['missing_information'].extend(missing)

            # Detect low confidence sections
            low_conf = await self._detect_low_confidence(content)
            weaknesses['low_confidence_sections'].extend(low_conf)

            # Detect potential hallucinations
            hallucinations = await self._detect_hallucinations(content, doc_name)
            weaknesses['potential_hallucinations'].extend(hallucinations)

            # Detect unsubstantiated claims
            unsubstantiated = await self._detect_unsubstantiated_claims(content)
            weaknesses['unsubstantiated_claims'].extend(unsubstantiated)

            # Detect vague statements
            vague = self._detect_vague_statements(content, doc_name)
            weaknesses['vague_statements'].extend(vague)

        # Check unaddressed requirements
        unaddressed = await self._check_requirements_coverage(
            documents, northstar_doc
        )
        weaknesses['unaddressed_requirements'] = unaddressed

        # Compile all issues
        weaknesses['all_issues'] = (
            weaknesses['missing_information'] +
            weaknesses['low_confidence_sections'] +
            weaknesses['potential_hallucinations'] +
            weaknesses['unsubstantiated_claims'] +
            weaknesses['vague_statements']
        )

        # Identify critical issues
        weaknesses['critical_issues'] = [
            issue for issue in weaknesses['all_issues']
            if issue.get('severity') == 'critical'
        ]

        # Calculate improvement potential
        weaknesses['improvement_potential'] = min(
            100,
            len(weaknesses['critical_issues']) * 10 +
            len(weaknesses['all_issues']) * 2
        )

        return weaknesses

    async def _detect_missing_information(
        self,
        content: str,
        northstar_doc: str,
        rfp_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect missing critical information"""

        prompt = f"""
        Analyze this document for missing critical information.

        DOCUMENT CONTENT: {content[:4000]}
        REQUIREMENTS: {northstar_doc[:2000]}
        RFP TYPE: {rfp_metadata.get('type', 'general')}

        Identify specific missing information that would strengthen this proposal:
        1. Missing technical details (tools, versions, methodologies)
        2. Absent pricing information (rates, cost breakdowns)
        3. Undefined team information (roles, clearances, certifications)
        4. Missing timelines (milestones, deliverables)
        5. Absent compliance details (certifications, standards)

        Return JSON:
        {{
            "gaps": [
                {{
                    "type": "technical|pricing|team|timeline|compliance",
                    "description": "What specific information is missing",
                    "severity": "critical|high|medium|low",
                    "section": "Where this should be addressed",
                    "suggested_question": "Specific question to ask user"
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
                max_tokens=1500
            )

            result = json.loads(response.choices[0].message.content)
            return result.get('gaps', [])

        except Exception as e:
            logger.error(f"Error detecting missing information: {e}")
            return []

    async def _detect_low_confidence(self, content: str) -> List[Dict[str, Any]]:
        """Detect sections with low confidence indicators"""

        low_confidence_sections = []

        # Split into paragraphs
        paragraphs = content.split('\n\n')

        for i, para in enumerate(paragraphs):
            if len(para) < 50:
                continue

            confidence_score = 1.0
            indicators = []

            # Check for uncertain language
            for pattern in self.weakness_indicators['uncertain_language']:
                matches = re.findall(pattern, para, re.IGNORECASE)
                if matches:
                    confidence_score -= 0.15 * len(matches)
                    indicators.extend(matches)

            # Check for vague claims
            for pattern in self.weakness_indicators['vague_claims']:
                if re.search(pattern, para, re.IGNORECASE):
                    confidence_score -= 0.2
                    indicators.append(pattern)

            if confidence_score < self.confidence_threshold:
                low_confidence_sections.append({
                    'paragraph_index': i,
                    'content_preview': para[:200],
                    'confidence_score': max(0, confidence_score),
                    'indicators': indicators,
                    'severity': 'high' if confidence_score < 0.5 else 'medium',
                    'suggested_question': f"Can you provide specific details for: {para[:100]}...?"
                })

        return low_confidence_sections

    async def _detect_hallucinations(
        self,
        content: str,
        doc_name: str
    ) -> List[Dict[str, Any]]:
        """Detect potential hallucinations or unsupported claims"""

        # Look for claims that might be hallucinated
        potential_hallucinations = []

        # Check for specific numeric claims without context
        numeric_claims = re.findall(
            r'(\d+(?:\.\d+)?%?\s*(?:improvement|increase|decrease|reduction|growth))',
            content,
            re.IGNORECASE
        )

        for claim in numeric_claims:
            # Check if claim has supporting context
            claim_context = self._get_context_around(content, claim, 100)
            if not any(word in claim_context.lower() for word in ['based on', 'according to', 'measured', 'verified']):
                potential_hallucinations.append({
                    'type': 'unsupported_metric',
                    'claim': claim,
                    'document': doc_name,
                    'severity': 'high',
                    'suggested_question': f"What is the source for this claim: {claim}?"
                })

        # Check for superlative claims
        superlatives = re.findall(
            r'(best|leading|top|premier|unmatched|superior|exceptional)',
            content,
            re.IGNORECASE
        )

        for superlative in set(superlatives):
            potential_hallucinations.append({
                'type': 'unverifiable_superlative',
                'claim': superlative,
                'document': doc_name,
                'severity': 'medium',
                'suggested_question': f"Can you provide evidence for being '{superlative}' in this context?"
            })

        return potential_hallucinations

    async def _detect_unsubstantiated_claims(self, content: str) -> List[Dict[str, Any]]:
        """Detect claims that lack substantiation"""

        unsubstantiated = []

        # Claims about experience without specifics
        experience_claims = re.findall(
            r'(\d+\+?\s*years?\s*(?:of\s*)?experience)',
            content,
            re.IGNORECASE
        )

        for claim in experience_claims:
            context = self._get_context_around(content, claim, 150)
            if not re.search(r'(project|contract|client|implementation)', context, re.IGNORECASE):
                unsubstantiated.append({
                    'type': 'experience_claim',
                    'claim': claim,
                    'severity': 'medium',
                    'suggested_question': f"Can you provide specific examples for: {claim}?"
                })

        return unsubstantiated

    def _detect_vague_statements(self, content: str, doc_name: str) -> List[Dict[str, Any]]:
        """Detect vague statements that need specificity"""

        vague_statements = []

        for pattern_type, patterns in self.weakness_indicators.items():
            if pattern_type == 'vague_claims' or pattern_type == 'missing_specifics':
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        vague_statements.append({
                            'type': 'vague_statement',
                            'statement': match.group(),
                            'document': doc_name,
                            'position': match.start(),
                            'severity': 'medium',
                            'suggested_question': f"Please provide specific details for: {match.group()}"
                        })

        return vague_statements

    async def _check_requirements_coverage(
        self,
        documents: Dict[str, str],
        northstar_doc: str
    ) -> List[Dict[str, Any]]:
        """Check if all requirements are addressed"""

        # Combine all documents
        all_content = "\n\n".join(documents.values())

        prompt = f"""
        Check if all requirements are addressed in the documents.

        REQUIREMENTS FROM NORTHSTAR: {northstar_doc[:3000]}
        COMBINED DOCUMENTS: {all_content[:5000]}

        Identify any requirements that are not adequately addressed.

        Return JSON:
        {{
            "unaddressed": [
                {{
                    "requirement": "Specific requirement text",
                    "severity": "critical|high|medium",
                    "suggested_question": "Question to get this information"
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
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)
            return result.get('unaddressed', [])

        except Exception as e:
            logger.error(f"Error checking requirements: {e}")
            return []

    async def generate_targeted_questions(
        self,
        weaknesses: Dict[str, Any],
        rfp_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific questions based on weaknesses"""

        questions = []
        question_id = 1

        # Process critical issues first
        for issue in weaknesses.get('critical_issues', []):
            questions.append({
                'id': f'Q{question_id:03d}',
                'type': issue.get('type', 'general'),
                'priority': 'critical',
                'question': issue.get('suggested_question', 'Please provide more information'),
                'context': issue.get('description', ''),
                'improvement_potential': 20
            })
            question_id += 1

        # Process missing information
        for gap in weaknesses.get('missing_information', [])[:5]:
            questions.append({
                'id': f'Q{question_id:03d}',
                'type': 'missing_info',
                'priority': gap.get('severity', 'medium'),
                'question': gap.get('suggested_question'),
                'context': gap.get('description'),
                'improvement_potential': 15
            })
            question_id += 1

        # Process hallucinations
        for hallucination in weaknesses.get('potential_hallucinations', [])[:3]:
            questions.append({
                'id': f'Q{question_id:03d}',
                'type': 'verification',
                'priority': 'high',
                'question': hallucination.get('suggested_question'),
                'context': f"Claim: {hallucination.get('claim')}",
                'improvement_potential': 10
            })
            question_id += 1

        # Process vague statements
        for vague in weaknesses.get('vague_statements', [])[:3]:
            questions.append({
                'id': f'Q{question_id:03d}',
                'type': 'clarification',
                'priority': 'medium',
                'question': vague.get('suggested_question'),
                'context': f"Statement: {vague.get('statement')}",
                'improvement_potential': 8
            })
            question_id += 1

        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        questions.sort(key=lambda q: priority_order.get(q['priority'], 999))

        return questions[:15]  # Return top 15 questions

    async def process_answer(
        self,
        session_id: str,
        question_id: str,
        answer: str
    ) -> Dict[str, Any]:
        """Process a human answer and potentially refine documents"""

        if session_id not in self.sessions:
            return {'error': 'Session not found'}

        session = self.sessions[session_id]

        # Find the question
        question = next((q for q in session.questions if q['id'] == question_id), None)
        if not question:
            return {'error': 'Question not found'}

        # Store the answered question
        session.answered_questions.append({
            'question_id': question_id,
            'question': question['question'],
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })

        # Extract knowledge from answer
        knowledge = await self._extract_knowledge(question, answer)

        # Determine if we should refine documents
        refinement = None
        if len(session.answered_questions) >= 3 or question['priority'] == 'critical':
            refinement = await self._create_refinement_suggestion(
                session.original_documents,
                session.answered_questions
            )
            session.refinements.append(refinement)

        # Get next questions
        remaining_questions = [
            q for q in session.questions
            if q['id'] not in [aq['question_id'] for aq in session.answered_questions]
        ]

        return {
            'session_id': session_id,
            'processed': True,
            'knowledge_extracted': knowledge,
            'refinement_suggested': refinement is not None,
            'refinement': refinement,
            'remaining_questions': len(remaining_questions),
            'next_questions': remaining_questions[:3]
        }

    async def _extract_knowledge(
        self,
        question: Dict[str, Any],
        answer: str
    ) -> Dict[str, Any]:
        """Extract reusable knowledge from human answer"""

        prompt = f"""
        Extract factual knowledge from this human response.

        QUESTION: {question['question']}
        CONTEXT: {question.get('context', '')}
        HUMAN ANSWER: {answer}

        Extract:
        1. Specific facts (numbers, names, tools, certifications)
        2. Clarifications of vague statements
        3. Evidence for claims
        4. Technical specifications

        Return JSON:
        {{
            "facts": [
                {{
                    "type": "technical_spec|pricing_info|compliance_detail|team_info",
                    "content": "The specific fact",
                    "confidence": 0.0-1.0,
                    "reusable": true/false
                }}
            ],
            "summary": "Brief summary of knowledge gained"
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}")
            return {'facts': [], 'summary': ''}

    async def _create_refinement_suggestion(
        self,
        original_documents: Dict[str, str],
        answered_questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create document refinement suggestions based on answers"""

        refinements = {}

        for qa in answered_questions:
            # Determine which document to refine
            # This is simplified - in practice would need better mapping
            for doc_name, content in original_documents.items():
                if qa['question'].lower() in content.lower()[:1000]:
                    if doc_name not in refinements:
                        refinements[doc_name] = []
                    refinements[doc_name].append({
                        'question': qa['question'],
                        'answer': qa['answer']
                    })
                    break

        return {
            'timestamp': datetime.now().isoformat(),
            'documents_to_refine': list(refinements.keys()),
            'refinement_count': sum(len(v) for v in refinements.values()),
            'refinements': refinements
        }

    def _get_context_around(self, text: str, target: str, context_size: int) -> str:
        """Get context around a target string"""

        index = text.find(target)
        if index == -1:
            return ""

        start = max(0, index - context_size)
        end = min(len(text), index + len(target) + context_size)

        return text[start:end]

    def _generate_session_id(self, notice_id: str) -> str:
        """Generate unique session ID"""

        timestamp = datetime.now().isoformat()
        data = f"{notice_id}_{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    async def save_conversation_to_rag(
        self,
        session_id: str
    ) -> bool:
        """Save valuable conversation data to RAG for future use"""

        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Extract all valuable facts from the conversation
        valuable_facts = []

        for qa in session.answered_questions:
            # Extract knowledge from each Q&A pair
            knowledge = await self._extract_knowledge(
                {'question': qa['question']},
                qa['answer']
            )

            for fact in knowledge.get('facts', []):
                if fact.get('reusable') and fact.get('confidence', 0) > 0.7:
                    valuable_facts.append({
                        'content': fact['content'],
                        'type': fact['type'],
                        'source': 'human_feedback',
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat()
                    })

        # Save to RAG if we have valuable facts
        if valuable_facts and self.milvus_rag:
            for fact in valuable_facts:
                # This would need to be implemented in your milvus_rag
                # await self.milvus_rag.add_knowledge(fact)
                logger.info(f"Would save to RAG: {fact['content'][:100]}")

        return len(valuable_facts) > 0

    async def start_refinement_session(
        self,
        notice_id: str,
        documents: Dict[str, str],
        review_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start a new refinement session"""
        session_id = self._generate_session_id(notice_id)

        # Analyze weaknesses
        weaknesses = await self.analyze_weakness_patterns(
            documents, None, {}
        )

        # Generate questions
        questions = await self.generate_targeted_questions(
            weaknesses, documents
        )

        # Create session
        session = {
            "session_id": session_id,
            "notice_id": notice_id,
            "created_at": datetime.now().isoformat(),
            "documents": documents,
            "initial_scores": review_result.get("overall_scores", {}),
            "review_result": review_result,
            "questions": questions,
            "answered_questions": [],
            "weaknesses": weaknesses,
            "status": "active"
        }

        # Store session
        self.sessions[session_id] = session

        return session

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a refinement session by ID"""
        return self.sessions.get(session_id)

    def _generate_session_id(self, notice_id: str) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{notice_id}_{timestamp}".encode()).hexdigest()[:16]
