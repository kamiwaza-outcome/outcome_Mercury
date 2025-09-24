import os
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from .openai_service import OpenAIService, CIRCUIT_BREAKER
import json
import asyncio
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pickle
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class OrchestrationAgent:
    def __init__(self, doc_generator=None, milvus_rag=None, mission_control=None):
        # Shared OpenAI client via OpenAIService to reduce sockets
        self.openai_service = OpenAIService()
        self.client = self.openai_service.sync_client
        self.doc_generator = doc_generator
        self.milvus_rag = milvus_rag
        self.mission_control = mission_control
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        # Use gpt-4o by default as the safer fallback for generation
        self.fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")
        self.skip_revision = os.getenv("SKIP_REVISION_PHASE", "false").lower() == "true"
        # Revision control parameters - based on LLM effectiveness research showing diminishing returns after 2-3 iterations
        self.max_revisions = int(os.getenv("MAX_REVISIONS", "2"))  # Optimal based on research
        self.revision_quality_threshold = float(os.getenv("REVISION_QUALITY_THRESHOLD", "85.0"))  # Minimum quality score to continue revisions
        # Token control for document generation
        # Cap completions to reduce timeouts and API stress
        self.max_completion_tokens = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30  # seconds
        # Streaming configuration for progress visibility
        # Streaming config: allow OPENAI_STREAMING override; disable by default for GPT-5
        default_streaming = "false" if "gpt-5" in self.model.lower() else "true"
        env_stream = os.getenv("OPENAI_STREAMING") or os.getenv("ENABLE_STREAMING", default_streaming)
        self.enable_streaming = (env_stream or default_streaming).lower() == "true"
        if not self.enable_streaming and "gpt-5" in self.model.lower():
            logger.info("Streaming disabled for GPT-5 model (requires organization verification)")
        # GPT-5 specific parameters
        # Shift to medium by default to balance latency & quality
        self.reasoning_effort = os.getenv("GPT5_REASONING_EFFORT", "medium")  # low, medium, high
        self.verbosity = os.getenv("GPT5_VERBOSITY", "medium")  # low, medium, high

    def _get_gpt5_params(self, model):
        """Get GPT-5 specific parameters if applicable"""
        if "gpt-5" in model.lower():
            return {
                "reasoning_effort": self.reasoning_effort,
                "verbosity": self.verbosity
            }
        return {}

    def _select_model(self, requested_model: str) -> str:
        """Apply circuit breaker for GPT-5 and return effective model."""
        if "gpt-5" in (requested_model or "").lower() and not CIRCUIT_BREAKER.should_allow():
            logger.warning("GPT-5 circuit open; using fallback model for orchestration")
            return self.fallback_model
        return requested_model

    def _get_token_param_name(self, model):
        """Get the correct token parameter name based on model"""
        # GPT-5 models and newer use max_completion_tokens
        if "gpt-5" in model.lower() or "gpt-4" in model.lower():
            return "max_completion_tokens"
        # Older models might use max_tokens
        return "max_tokens"

    def _create_completion_with_progress(self, model, messages, max_tokens, doc_name=None):
        """Create completion with streaming for progress visibility"""
        # Build parameters dict with correct token parameter name
        model = self._select_model(model)
        token_param = self._get_token_param_name(model)
        params = {
            "model": model,
            "messages": messages,
            token_param: max_tokens
        }

        # Add GPT-5 specific parameters if using GPT-5
        if "gpt-5" in model.lower():
            params["reasoning_effort"] = self.reasoning_effort
            params["verbosity"] = self.verbosity
            logger.info(f"Using GPT-5 with reasoning_effort={self.reasoning_effort}, verbosity={self.verbosity}")

        if self.enable_streaming and doc_name:
            try:
                logger.info(f"Generating {doc_name} with streaming enabled...")
                params["stream"] = True

                # Create streaming completion
                stream = self.client.chat.completions.create(**params)

                # Collect streamed response with progress logging
                full_response = ""
                chunk_count = 0
                last_log = time.time()

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        chunk_count += 1

                        # Log progress every 5 seconds
                        if time.time() - last_log > 5:
                            logger.info(f"{doc_name}: Generated {len(full_response)} characters so far...")
                            last_log = time.time()

                logger.info(f"{doc_name}: Generation complete - {len(full_response)} characters")
                if "gpt-5" in model.lower():
                    try:
                        CIRCUIT_BREAKER.record_success()
                    except Exception:
                        pass
                return full_response
            except Exception as e:
                # Check if it's a streaming verification error or organization not verified
                error_msg = str(e).lower()
                if ("verified to stream" in error_msg or
                    "stream" in error_msg or
                    "organization must be verified" in error_msg or
                    "unsupported_value" in error_msg):
                    logger.warning(f"Streaming not available for {model} (organization verification required), falling back to non-streaming mode: {e}")
                    # Retry without streaming
                    params.pop("stream", None)
                    try:
                        response = self.client.chat.completions.create(**params)
                        return response.choices[0].message.content
                    except Exception as inner_e:
                        # Check for GPT-5 parameter errors
                        inner_error_msg = str(inner_e).lower()
                        if ("reasoning_effort" in inner_error_msg or
                            "verbosity" in inner_error_msg or
                            "unsupported parameter" in inner_error_msg or
                            "extra_forbidden" in inner_error_msg):
                            logger.warning(f"GPT-5 parameters not supported for {model}, retrying without: {inner_e}")
                            params.pop("reasoning_effort", None)
                            params.pop("verbosity", None)
                            response = self.client.chat.completions.create(**params)
                            return response.choices[0].message.content
                        else:
                            raise
                elif ("reasoning_effort" in error_msg or
                      "verbosity" in error_msg or
                      "unsupported parameter" in error_msg or
                      "extra_forbidden" in error_msg):
                    logger.warning(f"GPT-5 parameters not supported for {model}, retrying without: {e}")
                    params.pop("reasoning_effort", None)
                    params.pop("verbosity", None)
                    params.pop("stream", None)  # Also remove stream if it was set
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content
                else:
                    # Re-raise other errors
                    raise
        else:
            # Non-streaming fallback
            try:
                response = self.client.chat.completions.create(**params)
                if "gpt-5" in model.lower():
                    try:
                        CIRCUIT_BREAKER.record_success()
                    except Exception:
                        pass
                return response.choices[0].message.content
            except Exception as e:
                # Check for GPT-5 parameter errors
                error_msg = str(e).lower()
                if ("reasoning_effort" in error_msg or
                    "verbosity" in error_msg or
                    "unsupported parameter" in error_msg or
                    "extra_forbidden" in error_msg):
                    logger.warning(f"GPT-5 parameters not supported for {model}, retrying without: {e}")
                    params.pop("reasoning_effort", None)
                    params.pop("verbosity", None)
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content
                else:
                    if "gpt-5" in model.lower():
                        try:
                            CIRCUIT_BREAKER.record_failure()
                        except Exception:
                            pass
                    raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def create_northstar_document(self, rfp_documents: Dict[str, Any], rfp_metadata: Dict[str, Any], company_context: str) -> str:
        """Create comprehensive Northstar document analyzing all RFP requirements"""
        try:
            # Log what RFP we're actually processing
            logger.info(f"Creating Northstar for RFP:")
            logger.info(f"  - Notice ID: {rfp_metadata.get('notice_id', 'Unknown')}")
            logger.info(f"  - Title: {rfp_metadata.get('title', 'Unknown')}")
            logger.info(f"  - Sol Number: {rfp_metadata.get('sol_number', 'Unknown')}")
            logger.info(f"  - Type: {rfp_metadata.get('type', 'Unknown')}")
            logger.info(f"  - NAICS: {rfp_metadata.get('naics', 'Unknown')}")
            
            all_content = self._combine_rfp_content(rfp_documents, rfp_metadata)
            
            # Log the first 500 chars of content to verify we have the right RFP
            logger.info(f"RFP content preview (first 500 chars): {all_content[:500] if all_content else 'No content'}")
            
            system_prompt = """You are an expert RFP analyst creating a comprehensive Northstar document that will drive ALL downstream decisions.
            Analyze ALL provided RFP documents, attachments, and information to create a detailed guide.
            
            CRITICAL: First, identify the SOLICITATION TYPE (RFI, RFP, RFQ, Sources Sought, etc.) as this determines the entire approach.
            
            Your Northstar document MUST include:
            
            1. EXECUTIVE SUMMARY
               - Solicitation type (RFI/RFP/RFQ/etc.) and what that means for our response
               - RFP title and ID  
               - Issuing agency
               - Key dates and deadlines
               - Brief scope overview
            
            2. COMPLETE REQUIREMENTS ANALYSIS
               - List EVERY requirement mentioned anywhere in the RFP
               - Technical requirements
               - Performance requirements
               - Compliance requirements
               - Documentation requirements
               - Formatting requirements
               - For each requirement, note confidence level (1-10) if ambiguous
               
            3. REQUIRED DELIVERABLES
               - List EVERY document that must be submitted
               - Format specifications for each
               - Page limits or size constraints
               - Submission method and naming conventions
            
            4. EVALUATION CRITERIA
               - How proposals will be scored
               - Weight of each section
               - Key evaluation factors
               - WHO evaluates WHAT (technical team vs contracting officer)
            
            5. WIN CONDITIONS
               - What will make our proposal stand out
               - Agency's pain points and priorities
               - Competitive advantages to emphasize
            
            6. SUBMISSION REQUIREMENTS
               - Exact submission process
               - Portal or email requirements
               - Registration requirements
               - Deadline (date and time with timezone)
            
            7. CRITICAL COMPLIANCE ITEMS
               - Mandatory certifications
               - Required forms
               - Security clearance requirements
               - Small business requirements
            
            8. DOCUMENT GENERATION STRATEGY
               - Based on solicitation type, what documents should we create?
               - How should requirements be grouped into documents?
               - What are the natural boundaries between documents?
               - Consider evaluation criteria: what will be reviewed together?
               - Confidence score (1-10) for document organization approach
            
            9. REQUIREMENT-TO-DOCUMENT MAPPING
               - For each major requirement category, specify target document
               - Explain consolidation decisions (why group certain items)
               - Flag any requirements that could go in multiple places
               - Note dependencies between requirements
            
            10. AMBIGUITY FLAGS & ASSUMPTIONS
               - List any unclear or contradictory requirements
               - State assumptions made for ambiguous items
               - Identify areas needing human clarification
               - Rate overall confidence in interpretation (1-10)
            
            Be EXTREMELY thorough - missing even one requirement could disqualify the proposal.
            Think about document organization intelligently - don't default to templates."""
            
            user_prompt = f"""Create a Northstar document for this RFP:
            
            RFP METADATA:
            {json.dumps(rfp_metadata, indent=2)}
            
            ALL RFP CONTENT AND ATTACHMENTS:
            {all_content}
            
            COMPANY CONTEXT:
            {company_context}
            
            Remember: Extract EVERY requirement, EVERY document needed, and EVERY evaluation criterion.
            Be exhaustive in your analysis."""
            
            # Heartbeat before API call
            self._heartbeat(f"Creating Northstar for {rfp_metadata.get('notice_id', 'Unknown')}")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=self.max_completion_tokens,
                    **self._get_gpt5_params(self.model)
                )
            except Exception as primary_err:
                logger.warning(f"Primary model {self.model} failed creating Northstar: {primary_err}. Falling back to {self.fallback_model}.")
                # Retry with fallback model and without GPT-5 params
                response = self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=self.max_completion_tokens
                )
            
            northstar = response.choices[0].message.content
            logger.info("Northstar document created successfully")
            return northstar
            
        except Exception as e:
            logger.error(f"Error creating Northstar document: {e}")
            raise
    
    async def generate_rfp_response(self, northstar_document: str, rfp_documents: Dict[str, Any], notice_id: str = None) -> Dict[str, str]:
        """Generate all required RFP response documents with checkpoint support"""
        try:
            # Try to load from checkpoint
            if notice_id:
                saved_docs = self._load_checkpoint(notice_id, "generated_documents")
                if saved_docs:
                    logger.info(f"Loaded {len(saved_docs)} documents from checkpoint")
                    return saved_docs
            
            required_documents = await self._identify_required_documents(northstar_document)
            
            logger.info(f"Identified {len(required_documents)} required documents")
            logger.info(f"Document type: {type(required_documents)}")
            logger.info(f"Documents content: {required_documents}")
            
            generated_docs = {}
            
            for doc_spec in required_documents:
                logger.info(f"Processing doc_spec: {doc_spec}, type: {type(doc_spec)}")
                doc_name = doc_spec['name']
                doc_requirements = doc_spec['requirements']
                doc_format = doc_spec.get('format', 'text')
                
                # Skip if already generated (from partial checkpoint)
                if doc_name in generated_docs:
                    logger.info(f"Skipping already generated document: {doc_name}")
                    continue
                
                logger.info(f"Generating document: {doc_name}")
                self._heartbeat(f"Generating {doc_name}")
                
                try:
                    if self._needs_specialized_agent(doc_spec):
                        content = await self._call_specialized_agent(
                            doc_name, doc_requirements, northstar_document
                        )
                    else:
                        content = await self._generate_document_content(
                            doc_name, doc_requirements, northstar_document
                        )
                    
                    generated_docs[doc_name] = content
                    
                    # Save checkpoint after each successful document
                    if notice_id:
                        self._save_checkpoint(notice_id, "generated_documents", generated_docs)
                        
                except Exception as e:
                    logger.error(f"Failed to generate {doc_name}: {e}")
                    # Continue with other documents instead of failing completely
                    generated_docs[doc_name] = f"ERROR: Failed to generate - {str(e)}"
            
            generated_docs = await self._run_evaluation_agents(generated_docs, northstar_document)
            
            return generated_docs
            
        except Exception as e:
            logger.error(f"Error generating RFP response: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30)
    )
    async def _identify_required_documents(self, northstar_document: str) -> List[Dict[str, Any]]:
        """Identify all documents that need to be created using intelligent reasoning"""
        try:
            prompt = f"""You are analyzing a Northstar document that contains comprehensive analysis of a government solicitation.
            Your task is to determine what documents should be created based on the ACTUAL requirements, not templates.
            
            NORTHSTAR DOCUMENT:
            {northstar_document}
            
            CRITICAL INSTRUCTIONS FOR INTELLIGENT DOCUMENT IDENTIFICATION:
            
            1. SELF-QUESTIONING APPROACH (ask yourself these questions):
               - What type of solicitation is this really? (RFI, RFQ, RFP, Sources Sought, etc.)
               - What does the government EXPLICITLY ask for in submission instructions?
               - Are there specific document names or formats mentioned?
               - What are the actual evaluation factors (if any)?
               - Is this asking for information/capabilities or a full proposal?
               - What would make logical sense given the solicitation's purpose?
            
            2. ADAPTIVE REASONING (no templates):
               - DO NOT default to Volume I-IV structure unless explicitly required
               - DO NOT create documents just because they're "standard"
               - ONLY create documents that directly address stated requirements
               - For RFIs: Usually 1-3 documents (capabilities statement, white paper, etc.)
               - For RFQs: Focus on pricing and minimal technical info
               - For RFPs: Follow their specific volume/section requirements
               - For Sources Sought: Usually just a capabilities statement
            
            3. USE THE NORTHSTAR'S GUIDANCE:
               - Look for "DOCUMENT GENERATION STRATEGY" section in the Northstar
               - Follow the "REQUIREMENT-TO-DOCUMENT MAPPING" if present
               - Consider any "AMBIGUITY FLAGS & ASSUMPTIONS" noted
               - Use the confidence scores to gauge certainty
            
            4. INTELLIGENT CONSOLIDATION:
               - Group related requirements that would be evaluated together
               - Combine items that support the same objective
               - Create fewer, comprehensive documents rather than many fragments
               - Consider: Would an evaluator want this as one document or separate?
            
            5. UNCERTAINTY SCORING:
               - Rate your confidence (1-10) for each document decision
               - Flag any assumptions you're making
               - Note where solicitation is ambiguous
               - Identify what might need human clarification
            
            Return a JSON object with this structure:
            {{
                "reasoning": {{
                    "solicitation_type": "Identified type and why",
                    "explicit_requirements": ["List of explicitly stated document requirements"],
                    "evaluation_factors": ["List of evaluation criteria if any"],
                    "key_assumptions": ["Assumptions made during analysis"],
                    "confidence_score": 8.5
                }},
                "documents": [
                    {{
                        "name": "document_name_with_extension",
                        "requirements": "detailed requirements and all content to include",
                        "format": "pdf/docx/xlsx/txt",
                        "page_limit": null or number,
                        "specialized": true/false,
                        "confidence": 9.0,
                        "reasoning": "Why this document is needed",
                        "consolidates": ["List of requirements consolidated into this doc"]
                    }}
                ]
            }}
            
            REMEMBER: 
            - This is NOT about following a template
            - Think critically about what's ACTUALLY being asked for
            - Use the Northstar's analysis to guide your decisions
            - When in doubt, consolidate rather than fragment
            - Quality over quantity - fewer, better documents"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an RFP requirements analyst. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                # temperature removed for GPT-5 compatibility
                # temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            logger.info(f"Raw response from GPT-5: {content[:1000]}")  # Log first 1000 chars
            
            # Handle different response formats from GPT-5
            try:
                # First try to parse the whole content as JSON
                parsed = json.loads(content)
                
                # Log reasoning if present (for intelligent document identification)
                if isinstance(parsed, dict) and 'reasoning' in parsed:
                    reasoning = parsed['reasoning']
                    logger.info(f"Document Identification Reasoning:")
                    logger.info(f"  Solicitation Type: {reasoning.get('solicitation_type', 'Unknown')}")
                    logger.info(f"  Explicit Requirements: {reasoning.get('explicit_requirements', [])}")
                    logger.info(f"  Confidence Score: {reasoning.get('confidence_score', 'N/A')}")
                    logger.info(f"  Key Assumptions: {reasoning.get('key_assumptions', [])}")
                
                # Check if it's a dict with various possible keys for documents array
                if isinstance(parsed, dict):
                    # Try different possible keys that GPT might use
                    for key in ['documents', 'required_documents', 'docs', 'list']:
                        if key in parsed and isinstance(parsed[key], list):
                            documents = parsed[key]
                            break
                    else:
                        # If no array found in dict, treat the dict as a single document
                        documents = [parsed]
                elif isinstance(parsed, list):
                    documents = parsed
                else:
                    # If it's a string or other type, wrap in array
                    documents = [{"name": str(parsed), "requirements": "Document", "format": "text"}]
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Direct JSON parse failed: {e}, attempting regex extraction")
                # Try to extract JSON array from text
                import re
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    try:
                        documents = json.loads(json_match.group())
                    except:
                        logger.error("Failed to parse extracted JSON")
                        raise ValueError("Could not parse JSON array from response")
                else:
                    # If no JSON found, return default documents
                    logger.warning("No JSON found in response, using defaults")
                    return [
                        {"name": "Technical_Proposal.docx", "requirements": "Technical approach and solution", "format": "docx"},
                        {"name": "Cost_Proposal.xlsx", "requirements": "Detailed cost breakdown", "format": "xlsx"},
                        {"name": "Executive_Summary.pdf", "requirements": "High-level overview", "format": "pdf"}
                    ]
            
            logger.info(f"Parsed {len(documents)} documents")
            
            # Ensure each document is a dict with required fields
            validated_docs = []
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    # If it's a string, convert to dict
                    validated_docs.append({
                        "name": doc,
                        "requirements": "Document requirements",
                        "format": "text"
                    })
                elif isinstance(doc, dict):
                    # Ensure required fields exist
                    if 'name' not in doc:
                        doc['name'] = f"Document_{i+1}.txt"
                    if 'requirements' not in doc:
                        doc['requirements'] = "Requirements not specified"
                    validated_docs.append(doc)
                else:
                    logger.warning(f"Skipping invalid document at index {i}: {doc}")
            
            if not validated_docs:
                logger.warning("No valid documents found, using defaults")
                return [
                    {"name": "Technical_Proposal.docx", "requirements": "Technical approach and solution", "format": "docx"},
                    {"name": "Cost_Proposal.xlsx", "requirements": "Detailed cost breakdown", "format": "xlsx"},
                    {"name": "Executive_Summary.pdf", "requirements": "High-level overview", "format": "pdf"}
                ]
            
            logger.info(f"Returning {len(validated_docs)} validated documents")
            return validated_docs
            
        except Exception as e:
            logger.error(f"Error identifying required documents: {e}")
            return [
                {"name": "Technical_Proposal.docx", "requirements": "Technical approach and solution", "format": "docx"},
                {"name": "Cost_Proposal.xlsx", "requirements": "Detailed cost breakdown", "format": "xlsx"},
                {"name": "Executive_Summary.pdf", "requirements": "High-level overview", "format": "pdf"}
            ]
    
    def _needs_specialized_agent(self, doc_spec: Dict[str, Any]) -> bool:
        """Determine if a document needs a specialized agent"""
        specialized_keywords = [
            'technical architecture', 'security', 'compliance', 'cost', 'budget',
            'pricing', 'legal', 'contract', 'terms', 'conditions', 'certification'
        ]
        
        doc_name_lower = doc_spec['name'].lower()
        requirements_lower = doc_spec.get('requirements', '').lower()
        
        for keyword in specialized_keywords:
            if keyword in doc_name_lower or keyword in requirements_lower:
                return True
        
        return doc_spec.get('specialized', False)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _call_specialized_agent(self, doc_name: str, requirements: str, northstar: str) -> str:
        """Call a specialized agent for specific document types"""
        try:
            # Get company context if milvus_rag is available
            company_context = ""
            past_rfps = []

            if self.milvus_rag:
                try:
                    company_context = await self.milvus_rag.get_company_context(requirements)
                    past_rfps = await self.milvus_rag.search_past_rfps(requirements)
                except Exception as e:
                    logger.warning(f"Could not get context from milvus_rag: {e}")
                    company_context = "Company information not available"
            else:
                company_context = "Company information not available"

            past_examples = "\n\n".join([rfp['content'] for rfp in past_rfps[:3]])

            # Special handling for video pitch documents
            if '.mp4' in doc_name.lower() or 'video' in doc_name.lower() or 'pitch' in doc_name.lower():
                agent_prompt = f"""You are creating a DETAILED VIDEO SCRIPT AND STORYBOARD for: {doc_name}

                This is a 5-minute video pitch that must be compelling, professional, and meet all requirements.

                REQUIREMENTS:
                {requirements}

                NORTHSTAR DOCUMENT:
                {northstar}

                COMPANY INFORMATION:
                {company_context}

                Create a COMPLETE video script/storyboard that includes:

                ## VIDEO SCRIPT STRUCTURE

                ### OPENING (0:00 - 0:30)
                [Visual descriptions, speaker script, on-screen text, graphics]

                ### PROBLEM IDENTIFICATION (0:30 - 1:30)
                [Detailed script addressing the DoD-relevant problem]

                ### SOLUTION DESCRIPTION (1:30 - 3:00)
                [Technical explanation with visuals, dependencies, risks, Focus Area alignment]

                ### DOD IMPACT (3:00 - 4:00)
                [Quantified benefits, case studies, comparisons]

                ### DIFFERENTIATION & PRICING (4:00 - 4:45)
                [Competitive advantages and pricing model]

                ### CLOSING (4:45 - 5:00)
                [Call to action, contact information]

                For each section include:
                - VISUAL: [Describe what appears on screen]
                - SCRIPT: [Exact words to be spoken]
                - TEXT OVERLAY: [Any text/graphics to display]
                - TRANSITIONS: [How to move between sections]

                Make it engaging, clear, and compelling for government evaluators."""
            else:
                agent_prompt = f"""You are a specialized agent creating: {doc_name}

                REQUIREMENTS:
                {requirements}

                NORTHSTAR DOCUMENT:
                {northstar}

                COMPANY INFORMATION:
                {company_context}

                SIMILAR PAST RESPONSES:
                {past_examples if past_examples else "No similar past responses found"}

                Create a comprehensive, compliant, and compelling document that:
                1. Addresses ALL requirements
                2. Highlights our competitive advantages
                3. Uses appropriate formatting and structure
                4. Includes specific details and examples
                5. Demonstrates understanding of the agency's needs

                Be specific, detailed, and professional."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert {doc_name.split('.')[0]} writer for government RFPs."},
                    {"role": "user", "content": agent_prompt}
                ],
                # temperature removed for GPT-5 compatibility
                # temperature=0.4,
                max_completion_tokens=self.max_completion_tokens,
                **self._get_gpt5_params(self.model)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error with specialized agent for {doc_name}: {e}")
            return await self._generate_document_content(doc_name, requirements, northstar)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _generate_document_content(self, doc_name: str, requirements: str, northstar: str) -> str:
        """Generate content for a document with intelligent self-questioning"""
        try:
            # Get company context if milvus_rag is available
            company_context = ""

            if self.milvus_rag:
                try:
                    company_context = await self.milvus_rag.get_company_context(requirements)
                except Exception as e:
                    logger.warning(f"Could not get context from milvus_rag: {e}")
                    company_context = "Company information not available"
            else:
                company_context = "Company information not available"
            
            prompt = f"""Generate content for: {doc_name}
            
            REQUIREMENTS:
            {requirements}
            
            NORTHSTAR GUIDANCE:
            {northstar}
            
            COMPANY INFORMATION:
            {company_context}
            
            CRITICAL SELF-QUESTIONING APPROACH:
            Before writing, ask yourself:
            1. What is the PRIMARY purpose of this document?
            2. Who will evaluate this and what are they looking for?
            3. What specific requirements must be addressed?
            4. Are there any ambiguities I need to resolve?
            5. What assumptions am I making?
            6. How confident am I in each section (1-10)?
            
            DOCUMENT GENERATION GUIDELINES:
            - Focus on addressing the ACTUAL requirements, not generic content
            - Use the Northstar's document generation strategy section
            - Include specific details from company information when relevant
            - Flag any areas where you lack information with [NEEDS CLARIFICATION]
            - Rate your confidence in meeting each requirement
            
            OUTPUT FORMAT:
            Start with a brief confidence assessment:
            [CONFIDENCE: X/10] [KEY ASSUMPTIONS: ...] [UNCERTAINTIES: ...]
            
            Then provide the actual document content that fully addresses all requirements.
            
            Remember: Quality over quantity. Be specific, be accurate, be relevant."""
            
            # Use streaming for better visibility
            content = self._create_completion_with_progress(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert RFP response writer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_completion_tokens,
                doc_name=doc_name
            )

            return content
            
        except Exception as e:
            logger.error(f"Error generating document content for {doc_name}: {e}")
            raise
    
    async def _run_evaluation_agents(self, documents: Dict[str, str], northstar: str) -> Dict[str, str]:
        """Run evaluation agents on generated documents with controlled revision cycles"""
        if self.skip_revision:
            logger.info("Skipping revision phase per configuration")
            return documents

        try:
            # Track revision cycles per document to prevent infinite loops and optimize quality
            revision_tracking = {doc_name: {"revisions": 0, "quality_scores": [], "initial_quality": 0}
                               for doc_name in documents.keys()}

            logger.info(f"Starting evaluation and revision process with max {self.max_revisions} revisions per document")

            # Iterative revision process with quality tracking
            for revision_cycle in range(self.max_revisions + 1):  # +1 for initial evaluation
                logger.info(f"Evaluation cycle {revision_cycle + 1}/{self.max_revisions + 1}")

                # Evaluate all documents in parallel
                evaluation_tasks = []
                documents_to_evaluate = []

                for doc_name, content in documents.items():
                    # Skip documents that have already reached max revisions or quality threshold
                    if (revision_tracking[doc_name]["revisions"] < self.max_revisions or revision_cycle == 0):
                        evaluation_tasks.append(
                            self._evaluate_document(doc_name, content, northstar,
                                                  revision_tracking[doc_name]["revisions"])
                        )
                        documents_to_evaluate.append(doc_name)

                if not evaluation_tasks:
                    logger.info("No documents need further evaluation")
                    break

                evaluations = await asyncio.gather(*evaluation_tasks)

                # Process evaluation results and determine revision needs
                documents_revised_this_cycle = []

                for i, (doc_name, evaluation) in enumerate(evaluations):
                    current_quality = evaluation.get('completeness_score', 0)
                    revision_tracking[doc_name]["quality_scores"].append(current_quality)

                    # Store initial quality for comparison
                    if revision_cycle == 0:
                        revision_tracking[doc_name]["initial_quality"] = current_quality

                    # Determine if revision is needed based on multiple factors
                    needs_revision = self._should_revise_document(
                        evaluation, revision_tracking[doc_name], revision_cycle
                    )

                    if needs_revision and revision_tracking[doc_name]["revisions"] < self.max_revisions:
                        # Get issues from evaluation - try both 'issues' and 'critical_issues' keys
                        issues_to_fix = evaluation.get('issues', evaluation.get('critical_issues', []))

                        logger.info(
                            f"Revising {doc_name} (attempt {revision_tracking[doc_name]['revisions'] + 1}/{self.max_revisions})"
                            f" - Quality: {current_quality:.1f}%, Issues: {len(issues_to_fix)}"
                        )

                        # Perform revision
                        revised_content = await self._revise_document(
                            doc_name, documents[doc_name], issues_to_fix,
                            northstar, revision_tracking[doc_name]["revisions"]
                        )

                        documents[doc_name] = revised_content
                        revision_tracking[doc_name]["revisions"] += 1
                        documents_revised_this_cycle.append(doc_name)

                        # Heartbeat for long-running revisions
                        self._heartbeat(f"Revised {doc_name} - cycle {revision_cycle + 1}")

                    elif revision_tracking[doc_name]["revisions"] >= self.max_revisions:
                        logger.info(
                            f"Document {doc_name} reached maximum revisions ({self.max_revisions}). "
                            f"Final quality: {current_quality:.1f}%"
                        )
                    else:
                        logger.info(
                            f"Document {doc_name} meets quality standards - Quality: {current_quality:.1f}%"
                        )

                # If no documents were revised this cycle, we're done
                if not documents_revised_this_cycle:
                    logger.info(f"No documents revised in cycle {revision_cycle + 1}, evaluation complete")
                    break

            # Log final revision statistics
            self._log_revision_metrics(revision_tracking)

            return documents

        except Exception as e:
            logger.error(f"Error running evaluation agents: {e}")
            return documents
    
    async def _evaluate_document(self, doc_name: str, content: str, northstar: str, revision_count: int = 0) -> tuple:
        """Evaluate a single document with revision history awareness"""
        try:
            # Adjust evaluation criteria based on revision count (diminishing returns)
            revision_context = ""
            if revision_count > 0:
                revision_context = f"""
            REVISION CONTEXT:
            - This is revision attempt #{revision_count + 1}
            - Focus on identifying only CRITICAL issues that significantly impact compliance or quality
            - Consider diminishing returns: minor improvements may not justify additional revisions
            - Be more lenient with subjective quality issues in later revisions
            """

            # Use more content for later revisions to catch issues missed in earlier truncated evaluations
            content_limit = 3000 + (revision_count * 1000)  # Increase content reviewed in later revisions
            content_sample = content[:content_limit] + "..." if len(content) > content_limit else content

            prompt = f"""Evaluate this government RFP response document for compliance and quality.

            DOCUMENT: {doc_name}
            CONTENT: {content_sample}

            NORTHSTAR REQUIREMENTS:
            {northstar[:2500]}...
            {revision_context}

            EVALUATION CRITERIA (prioritized by importance):
            1. CRITICAL: Compliance with mandatory requirements (deal-breakers)
            2. HIGH: Completeness of required sections and information
            3. MEDIUM: Professional quality and clarity
            4. LOW: Style and minor formatting issues

            SCORING GUIDELINES:
            - 90-100: Excellent, ready for submission
            - 80-89: Good quality, minor improvements could help
            - 70-79: Acceptable, moderate issues present
            - 60-69: Below standard, significant improvements needed
            - Below 60: Major issues, revision required

            Return JSON:
            {{
                "compliant": true/false,
                "completeness_score": 0-100,
                "quality_tier": "excellent|good|acceptable|below_standard|poor",
                "needs_revision": true/false,
                "critical_issues": ["issues that must be fixed"],
                "improvement_opportunities": ["non-critical improvements"],
                "strengths": ["document strengths"],
                "confidence": 0-100
            }}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an RFP compliance evaluator. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                # temperature removed for GPT-5 compatibility
                # temperature=0.2,
                response_format={"type": "json_object"},
                max_completion_tokens=self.max_completion_tokens,
                **self._get_gpt5_params(self.model)
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return (doc_name, evaluation)
            
        except Exception as e:
            logger.error(f"Error evaluating {doc_name}: {e}")
            return (doc_name, {"needs_revision": False})
    
    async def _revise_document(self, doc_name: str, content: str, issues: List[str], northstar: str, revision_count: int = 0) -> str:
        """Revise a document based on evaluation feedback with revision history awareness"""
        try:
            # Customize revision approach based on iteration count
            revision_guidance = ""
            if revision_count == 0:
                revision_guidance = "FIRST REVISION: Focus on comprehensive improvements addressing all major issues."
            elif revision_count == 1:
                revision_guidance = "FINAL REVISION: Make targeted, high-impact changes only. Avoid over-editing."
            else:
                revision_guidance = "MAXIMUM REVISIONS REACHED: This should not occur in normal operation."

            prompt = f"""Revise this government RFP response document based on evaluation feedback.

            DOCUMENT: {doc_name}
            REVISION ATTEMPT: #{revision_count + 1}
            {revision_guidance}

            CURRENT CONTENT:
            {content}

            CRITICAL ISSUES TO ADDRESS:
            {json.dumps(issues, indent=2)}

            NORTHSTAR REQUIREMENTS:
            {northstar[:2500]}...

            REVISION GUIDELINES:
            1. Address ALL critical compliance issues first
            2. Maintain existing strengths and structure where possible
            3. Make targeted improvements rather than wholesale rewrites
            4. Ensure professional tone and clarity
            5. Preserve any company-specific information and context

            IMPORTANT:
            - Return the complete revised document
            - Focus on substance over style changes
            - Maintain document length and structure appropriateness
            - Do not add placeholder text or [NEEDS CLARIFICATION] tags

            Provide the complete revised document:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document editor."},
                    {"role": "user", "content": prompt}
                ],
                # temperature removed for GPT-5 compatibility
                # temperature=0.3,
                max_completion_tokens=self.max_completion_tokens,
                **self._get_gpt5_params(self.model)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error revising {doc_name}: {e}")
            return content

    def _should_revise_document(self, evaluation: Dict[str, Any], revision_tracking: Dict[str, Any], revision_cycle: int) -> bool:
        """Determine if a document should be revised based on evaluation and revision history"""
        try:
            # Extract evaluation metrics
            quality_score = evaluation.get('completeness_score', 0)
            is_compliant = evaluation.get('compliant', False)
            critical_issues = evaluation.get('critical_issues', evaluation.get('issues', []))
            quality_tier = evaluation.get('quality_tier', 'poor')

            # Never revise if already at max revisions
            if revision_tracking['revisions'] >= self.max_revisions:
                return False

            # Skip revision if environment says to skip
            if self.skip_revision:
                return False

            # Always revise critical compliance issues (unless at max revisions)
            if critical_issues and len(critical_issues) > 0:
                if revision_cycle == 0:  # First evaluation
                    logger.info(f"Revision needed: {len(critical_issues)} critical issues found")
                    return True
                else:
                    # In later cycles, only revise for critical issues that impact compliance
                    compliance_critical = any(
                        keyword in str(critical_issues).lower()
                        for keyword in ['compliance', 'requirement', 'mandatory', 'must', 'shall']
                    )
                    if compliance_critical:
                        logger.info(f"Revision needed: Critical compliance issues remain")
                        return True

            # Quality-based revision decisions
            if revision_cycle == 0:  # First evaluation - be more generous with revisions
                if quality_score < self.revision_quality_threshold:
                    logger.info(f"Revision needed: Quality score {quality_score:.1f}% below threshold {self.revision_quality_threshold}%")
                    return True
                elif quality_tier in ['poor', 'below_standard']:
                    logger.info(f"Revision needed: Quality tier '{quality_tier}' requires improvement")
                    return True
            else:  # Later evaluations - stricter criteria due to diminishing returns
                if quality_score < (self.revision_quality_threshold - 10):  # Lower bar for later revisions
                    logger.info(f"Revision needed: Quality score {quality_score:.1f}% significantly below threshold")
                    return True
                elif quality_tier == 'poor':  # Only revise if truly poor
                    logger.info(f"Revision needed: Document quality remains poor")
                    return True

            # Check for quality improvement trend
            if len(revision_tracking['quality_scores']) > 1:
                quality_improvement = quality_score - revision_tracking['quality_scores'][-2]
                if quality_improvement < 5 and revision_cycle > 0:  # Diminishing returns
                    logger.info(f"Skipping revision: Quality improvement plateauing ({quality_improvement:.1f}%)")
                    return False

            # Default to not revising if criteria not met
            logger.info(f"No revision needed: Quality {quality_score:.1f}%, Tier: {quality_tier}")
            return False

        except Exception as e:
            logger.error(f"Error determining revision need: {e}")
            return False

    def _log_revision_metrics(self, revision_tracking: Dict[str, Dict[str, Any]]) -> None:
        """Log comprehensive revision metrics for analysis and optimization"""
        try:
            total_documents = len(revision_tracking)
            total_revisions = sum(doc['revisions'] for doc in revision_tracking.values())

            logger.info("=== REVISION CYCLE METRICS ===")
            logger.info(f"Documents processed: {total_documents}")
            logger.info(f"Total revisions performed: {total_revisions}")
            logger.info(f"Average revisions per document: {total_revisions / total_documents:.1f}")

            # Per-document metrics
            for doc_name, tracking in revision_tracking.items():
                initial_quality = tracking['initial_quality']
                final_quality = tracking['quality_scores'][-1] if tracking['quality_scores'] else initial_quality
                quality_improvement = final_quality - initial_quality

                logger.info(
                    f"  {doc_name}: {tracking['revisions']} revisions, "
                    f"Quality: {initial_quality:.1f}% → {final_quality:.1f}% "
                    f"(+{quality_improvement:+.1f}%)"
                )

            # System performance metrics
            documents_improved = sum(
                1 for tracking in revision_tracking.values()
                if len(tracking['quality_scores']) > 1 and
                   tracking['quality_scores'][-1] > tracking['quality_scores'][0]
            )

            documents_max_revisions = sum(
                1 for tracking in revision_tracking.values()
                if tracking['revisions'] >= self.max_revisions
            )

            logger.info(f"Documents improved through revision: {documents_improved}/{total_documents}")
            logger.info(f"Documents reaching max revisions: {documents_max_revisions}/{total_documents}")

            # Efficiency metrics
            avg_quality_improvement = sum(
                tracking['quality_scores'][-1] - tracking['initial_quality']
                for tracking in revision_tracking.values()
                if tracking['quality_scores']
            ) / total_documents if total_documents > 0 else 0

            logger.info(f"Average quality improvement: {avg_quality_improvement:+.1f}%")
            logger.info("=== END REVISION METRICS ===")

        except Exception as e:
            logger.error(f"Error logging revision metrics: {e}")
    
    async def generate_review_guide(self, generated_documents: Dict[str, str], northstar_document: str) -> str:
        """Generate human review guide"""
        try:
            all_docs_summary = "\n\n".join([
                f"Document: {name}\nFirst 500 chars: {content[:500]}..."
                for name, content in generated_documents.items()
            ])
            
            prompt = f"""Create a comprehensive human review guide for these RFP response documents.
            
            GENERATED DOCUMENTS:
            {all_docs_summary}
            
            NORTHSTAR DOCUMENT:
            {northstar_document}
            
            Create a guide that includes:
            
            1. EXECUTIVE SUMMARY
               - What was generated
               - Overall confidence level
               - Critical items needing review
            
            2. DOCUMENT-BY-DOCUMENT REVIEW
               For each document:
               - What was successfully completed
               - Areas of uncertainty (BE SPECIFIC)
               - Missing information that needs human input
               - Sections that could be strengthened
            
            3. CRITICAL REVIEW ITEMS
               - High-risk areas that MUST be reviewed
               - Compliance concerns
               - Technical accuracy questions
               - Cost/pricing validations needed
            
            4. SUBMISSION CHECKLIST
               - [ ] All required documents present
               - [ ] Format requirements met
               - [ ] Page limits observed
               - [ ] Naming conventions followed
               - [ ] Registration completed
               - [ ] Portal access confirmed
               
            5. HUMAN COMPLETION TASKS
               Step-by-step instructions for:
               - Information that couldn't be auto-filled
               - Final quality checks
               - Submission process
               - Deadline reminders
            
            Be honest about limitations and very specific about what needs human attention."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are creating a review guide for humans to finalize an RFP response."},
                    {"role": "user", "content": prompt}
                ],
                # temperature removed for GPT-5 compatibility
                # temperature=0.3,
                max_completion_tokens=self.max_completion_tokens,
                **self._get_gpt5_params(self.model)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating review guide: {e}")
            raise
    
    def _combine_rfp_content(self, documents: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Combine all RFP content into a single string with token-aware sizing"""
        import hashlib
        import tiktoken
        
        # Use GPT-5's tokenizer (or fallback to GPT-4's)
        try:
            encoding = tiktoken.encoding_for_model("gpt-5")
        except:
            encoding = tiktoken.encoding_for_model("gpt-4")
        
        # GPT-5 has 400k token context window, reserve some for output
        MAX_INPUT_TOKENS = 250000  # Conservative limit for input
        
        combined = f"METADATA:\n{json.dumps(metadata, indent=2)}\n\n"
        current_tokens = len(encoding.encode(combined))
        
        # Track content integrity
        content_hashes = {}
        total_original_size = 0
        total_included_size = 0
        
        for doc_name, content in documents.items():
            doc_header = f"\n\n--- DOCUMENT: {doc_name} ---\n"
            
            if isinstance(content, str):
                # Calculate content hash for integrity tracking
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
                content_hashes[doc_name] = content_hash
                total_original_size += len(content)
                
                # Check if we can fit the entire document
                doc_tokens = len(encoding.encode(doc_header + content))
                
                if current_tokens + doc_tokens <= MAX_INPUT_TOKENS:
                    # Include full document
                    combined += doc_header + content
                    current_tokens += doc_tokens
                    total_included_size += len(content)
                    logger.info(f"Included full document {doc_name}: {len(content)} chars, {doc_tokens} tokens, hash: {content_hash}")
                else:
                    # Include what we can fit
                    remaining_tokens = MAX_INPUT_TOKENS - current_tokens - len(encoding.encode(doc_header))
                    if remaining_tokens > 1000:  # Only include if we have meaningful space
                        # Decode tokens to get approximate character count
                        truncated_content = encoding.decode(encoding.encode(content)[:remaining_tokens])
                        combined += doc_header + truncated_content + "\n[CONTENT TRUNCATED DUE TO TOKEN LIMIT]"
                        total_included_size += len(truncated_content)
                        logger.warning(f"Truncated document {doc_name}: {len(content)} -> {len(truncated_content)} chars, hash: {content_hash}")
                    else:
                        logger.warning(f"Skipped document {doc_name}: No space remaining in context window")
                    break  # Stop processing more documents
            else:
                content_str = str(content)
                combined += doc_header + content_str
                current_tokens += len(encoding.encode(doc_header + content_str))
        
        # Log content integrity information
        logger.info(f"Content processing complete: {total_included_size}/{total_original_size} chars included, {current_tokens} tokens used")
        if content_hashes:
            logger.info(f"Content hashes for verification: {content_hashes}")
        
        return combined
    
    def _heartbeat(self, message: str):
        """Log heartbeat message to show process is alive"""
        current_time = time.time()
        if current_time - self.last_heartbeat >= self.heartbeat_interval:
            logger.info(f"[HEARTBEAT] {message} - Process alive at {datetime.now().isoformat()}")
            self.last_heartbeat = current_time
    
    def _save_checkpoint(self, notice_id: str, checkpoint_name: str, data: Any):
        """Save checkpoint data for resume capability"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{notice_id}_{checkpoint_name}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, notice_id: str, checkpoint_name: str) -> Any:
        """Load checkpoint data if exists"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{notice_id}_{checkpoint_name}.pkl"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Checkpoint loaded: {checkpoint_file}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        return None
    
    def clear_checkpoints(self, notice_id: str):
        """Clear all checkpoints for a notice"""
        try:
            for checkpoint_file in self.checkpoint_dir.glob(f"{notice_id}_*.pkl"):
                checkpoint_file.unlink()
                logger.info(f"Cleared checkpoint: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to clear checkpoints: {e}")
    async def process_with_improvements(self, notice_id: str, improvement_context: str,
                                       original_documents: Dict[str, Any],
                                       feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process RFP with improvements based on human feedback"""
        try:
            logger.info(f"Processing improvements for {notice_id} based on feedback")

            # Log improvement start if mission control available
            if self.mission_control:
                await self.mission_control.log_event(
                    session_id=f"improve_{notice_id}",
                    event_type="improvement_started",
                    message="Starting document improvements based on review feedback",
                    data={"notice_id": notice_id}
                )

            # Generate improved documents
            improved_docs = {}

            for doc_name, original_content in original_documents.items():
                logger.info(f"Improving {doc_name}")

                improvement_prompt = f"""
                Based on the following human feedback and improvement instructions,
                revise and improve the {doc_name} document.

                ORIGINAL DOCUMENT:
                {original_content[:8000]}  # Limit for context

                IMPROVEMENT INSTRUCTIONS:
                {improvement_context}

                SPECIFIC FEEDBACK:
                {json.dumps(feedback_data.get('answers', {}), indent=2)}

                Generate an improved version that:
                1. Incorporates all feedback provided
                2. Strengthens weak areas identified
                3. Adds specific details mentioned in feedback
                4. Ensures full compliance with requirements
                5. Maintains professional tone and structure

                Return the complete improved document.
                """

                improved_content = self._create_completion_with_progress(
                    model=self.model,
                    messages=[{"role": "user", "content": improvement_prompt}],
                    max_tokens=self.max_completion_tokens,
                    doc_name=f"Improved {doc_name}"
                )

                improved_docs[doc_name] = improved_content

                if self.mission_control:
                    await self.mission_control.log_event(
                        session_id=f"improve_{notice_id}",
                        event_type="document_improved",
                        message=f"Improved {doc_name}",
                        data={"document": doc_name}
                    )

            # Save improved documents
            if self.doc_generator:
                await self.doc_generator.save_improved_documents(notice_id, improved_docs)

            if self.mission_control:
                await self.mission_control.log_event(
                    session_id=f"improve_{notice_id}",
                    event_type="improvement_completed",
                    message="All documents improved based on feedback",
                    data={
                        "notice_id": notice_id,
                        "documents_improved": list(improved_docs.keys())
                    }
                )

            return {
                "status": "improved",
                "documents": improved_docs,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing improvements: {e}")
            raise
