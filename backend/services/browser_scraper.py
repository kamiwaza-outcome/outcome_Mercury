import os
import logging
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import tempfile
import json
import hashlib
import re
import aiohttp
from browser_use import Agent, Browser, ChatOpenAI
import pypdf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class BrowserScraper:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # Use faster model for extraction
        self.download_dir = Path(tempfile.mkdtemp(prefix="rfp_downloads_"))
        
    async def scrape_rfp(self, url: str, notice_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Scrape RFP details and documents using Browser Use with GPT-5"""
        try:
            logger.info(f"Starting Browser Use scraping for {notice_id}")
            logger.info(f"URL: {url}")
            logger.info(f"Notice ID: {notice_id}")
            
            # Initialize the LLM for Browser Use with optimized settings
            # Note: Browser Use library expects max_tokens, not max_completion_tokens
            llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=0.1,  # Lower for consistent behavior
                timeout=300,  # 5 minutes timeout for LLM calls
                max_tokens=4096  # Browser Use library expects max_tokens parameter
            )
            
            # Create headless browser with optimized settings
            browser = Browser(
                headless=True,
                args=[  # Correct parameter name for Browser v0.7.3
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",  # Help with cross-origin issues
                    "--disable-features=IsolateOrigins,site-per-process"  # Help with iframe access
                ]
            )
            
            # Create Browser Use agent with browser-use v0.7.3 with proper configuration
            agent = Agent(
                task=(
                    f"Navigate to {url} and extract ALL content from this RFP page.\n"
                    "\n"
                    "Instructions:\n"
                    "1. Wait for the page to fully load (10 seconds)\n"
                    "2. Extract all visible text content from the entire page\n"
                    "3. Scroll down to see more content if needed\n"
                    "4. Click on 'Attachments' or 'Documents' or 'Links' section if visible\n"
                    "5. Extract ALL attachment download URLs - look for links with patterns like:\n"
                    "   - /api/opportunities/v2/download/\n"
                    "   - /api/prod/opps/v3/opportunities/*/resources/download/\n"
                    "   - Any href links to PDF, DOCX, XLSX files\n"
                    "6. Return the complete extracted text content INCLUDING a section titled 'ATTACHMENT_URLS:' with all download links\n"
                    "\n"
                    "Focus on extracting the actual page content and attachment URLs.\n"
                    "Continue until you have extracted all sections and attachment links."
                ),
                llm=llm,
                browser=browser,
                use_vision=False,  # Disable vision for speed
                max_steps=20,  # Increase to allow complete extraction
                max_failures=3,  # Keep retries
                max_actions_per_step=5,  # Keep action limit
                save_conversation_path=f"logs/browser_use_{notice_id}.json"  # Enable debugging
            )
            
            # Run the agent with smart timeout handling
            import asyncio
            try:
                # Extended timeout for complete extraction (5 minutes)
                result = await asyncio.wait_for(agent.run(), timeout=300)
                logger.info(f"Browser Use completed successfully in under 5 minutes")
            except asyncio.TimeoutError:
                logger.info(f"Browser Use timed out after 300s (5 minutes), checking partial results")
                # Get partial results if available
                if hasattr(agent, 'history'):
                    result = agent.history
                    logger.info(f"Retrieved partial results from agent history")
                else:
                    result = None
                    logger.warning(f"No partial results available from Browser Use")
            
            # Process the agent's result
            documents = {}
            rfp_metadata = {
                'notice_id': notice_id,
                'url': url,
                'scraped_data': {}
            }
            
            # Extract content from AgentHistoryList result
            if result:
                # Try multiple methods to extract content from browser-use result
                all_extracted_content = None
                final_result = None
                
                # Method 1: Try extracted_content() method
                try:
                    if hasattr(result, 'extracted_content'):
                        all_extracted_content = result.extracted_content()
                        logger.info(f"extracted_content() returned {type(all_extracted_content)}")
                except Exception as e:
                    logger.warning(f"Could not call extracted_content(): {e}")
                
                # Method 2: Try final_result() method  
                try:
                    if hasattr(result, 'final_result'):
                        final_result = result.final_result()
                        logger.info(f"final_result() returned {type(final_result)}")
                except Exception as e:
                    logger.warning(f"Could not call final_result(): {e}")
                
                # Method 3: Try to access action_results if other methods failed
                if not final_result and not all_extracted_content:
                    try:
                        if hasattr(result, 'all_results'):
                            # Access the all_results list directly
                            for action_result in result.all_results:
                                if hasattr(action_result, 'extracted_content') and action_result.extracted_content:
                                    content = action_result.extracted_content
                                    if content and len(str(content)) > 50:
                                        if not documents:
                                            documents['main_content.txt'] = str(content)
                                        else:
                                            documents[f'action_content_{len(documents)}.txt'] = str(content)
                                        logger.info(f"Extracted from action_result: {len(str(content))} chars")
                    except Exception as e:
                        logger.warning(f"Could not access all_results: {e}")
                
                # Store the final extracted content as main content
                if final_result:
                    documents['main_content.txt'] = str(final_result)
                    logger.info(f"Extracted final result content: {len(str(final_result))} characters")
                elif not documents:
                    # Fallback: convert the whole result to string if nothing else worked
                    logger.warning("Using fallback: converting entire result to string")
                    documents['main_content.txt'] = str(result)
                
                # Store all extracted content pieces if there are multiple
                if all_extracted_content:
                    try:
                        for i, content in enumerate(all_extracted_content):
                            if content and str(content).strip():
                                documents[f'extracted_content_{i}.txt'] = str(content)
                                logger.info(f"Extracted content piece {i}: {len(str(content))} characters")
                    except Exception as e:
                        logger.warning(f"Error processing extracted_content list: {e}")
                
                # Try to parse structured data from the final result
                try:
                    content_to_parse = final_result or (all_extracted_content[0] if all_extracted_content else '')
                    if content_to_parse and isinstance(content_to_parse, str) and ('{' in content_to_parse):
                        # Try to extract JSON from the result
                        json_start = content_to_parse.find('{')
                        json_end = content_to_parse.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = content_to_parse[json_start:json_end]
                            rfp_metadata['scraped_data'] = json.loads(json_str)
                            logger.info("Successfully parsed JSON from extracted content")
                except Exception as e:
                    logger.warning(f"Could not parse JSON from extracted content: {e}")
                    # Store raw content as fallback
                    content_to_store = final_result or str(result)
                    rfp_metadata['scraped_data'] = {'raw_content': content_to_store}
                
                # Also look for any files browser-use might have written to disk
                current_dir = Path.cwd()
                for pattern in ['extracted_content_*.md', 'extracted_content*.txt', 'todo.md']:
                    for file_path in current_dir.glob(pattern):
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            # Always capture these files - they contain our extracted data
                            documents[f'{file_path.name}'] = content
                            logger.info(f"Found and captured browser-use file: {file_path.name} ({len(content)} chars)")
                        except Exception as e:
                            logger.warning(f"Could not read browser-use file {file_path}: {e}")
                
                logger.info(f"Total extracted documents: {len(documents)}")
            else:
                logger.warning("Browser-use agent returned no result")
            
            # Check for browser-use created files in current working directory
            try:
                import glob
                cwd = Path.cwd()
                
                # Look for extracted_content files created by browser-use
                for pattern in ['extracted_content*.md', 'extracted_content*.txt', 'todo.md']:
                    for file_path in glob.glob(str(cwd / pattern)):
                        try:
                            file_path = Path(file_path)
                            if file_path.exists():
                                content = file_path.read_text(encoding='utf-8', errors='ignore')
                                if content and len(content) > 50:
                                    file_key = f"browseruse_{file_path.name}"
                                    documents[file_key] = content
                                    logger.info(f"Read browser-use file {file_path.name}: {len(content)} chars")
                                # Keep the file for debugging (don't delete)
                                # file_path.unlink()
                                logger.info(f"Keeping browser-use file {file_path.name} for debugging")
                        except Exception as e:
                            logger.debug(f"Error reading {file_path}: {e}")
            except Exception as e:
                logger.debug(f"Error checking for browser-use files: {e}")
            
            # Check if Browser Use downloaded any files
            download_path = Path.home() / "Downloads"  # Default download directory
            temp_dirs = [download_path, Path("/tmp")]
            
            # Also check /tmp/browser-use-* directories
            try:
                import glob
                for tmp_dir in glob.glob("/tmp/browser-use-*"):
                    temp_dirs.append(Path(tmp_dir))
            except:
                pass
            
            # Look for recently downloaded files
            import time
            current_time = time.time()
            
            for search_dir in temp_dirs:
                if not search_dir.exists():
                    continue
                    
                for file_pattern in ['*.pdf', '*.docx', '*.doc', '*.xlsx', '*.xls', '*.zip', '*.txt']:
                    try:
                        for file_path in glob.glob(str(search_dir / file_pattern)):
                            file_path = Path(file_path)
                            # Check if file was modified in the last 120 seconds (increased from 60)
                            if (current_time - file_path.stat().st_mtime) < 120:
                                try:
                                    # Read the file content
                                    if file_path.suffix == '.pdf':
                                        content = self._extract_pdf_text(file_path)
                                    elif file_path.suffix in ['.txt', '.md']:
                                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                                    elif file_path.suffix in ['.docx', '.doc']:
                                        content = self._extract_docx_text(file_path)
                                    else:
                                        content = f"Binary file downloaded: {file_path.name}"
                                    
                                    if file_path.name not in documents:
                                        documents[file_path.name] = content
                                        logger.info(f"Captured downloaded file: {file_path.name} from {search_dir}")
                                except Exception as e:
                                    logger.warning(f"Error processing downloaded file {file_path}: {e}")
                    except Exception as e:
                        logger.debug(f"Error scanning {search_dir} for {file_pattern}: {e}")
            
            # Content validation with checksums
            content_validation = {}
            total_content_size = 0
            valid_documents = {}
            
            for doc_name, content in documents.items():
                if content and isinstance(content, str) and len(content.strip()) > 20:
                    # Check if this is Browser Use action logs instead of real content
                    is_action_log = any([
                        'AgentHistoryList' in content,
                        'ActionResult(' in content,
                        'all_results=[' in content,
                        content.strip().startswith('🔗 Navigated to'),
                        content.strip() == 'Waited for 12 seconds',
                        'is_done=False' in content and 'success=None' in content
                    ])
                    
                    if is_action_log:
                        logger.warning(f"Detected Browser Use action logs in {doc_name}, not real content")
                        continue
                    
                    # Calculate content hash for integrity
                    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                    content_validation[doc_name] = {
                        'size': len(content),
                        'hash': content_hash,
                        'first_100_chars': content[:100].replace('\n', ' ')
                    }
                    total_content_size += len(content)
                    valid_documents[doc_name] = content
                    logger.info(f"Validated document {doc_name}: {len(content)} chars, hash: {content_hash}")
                else:
                    logger.warning(f"Skipping invalid/empty document {doc_name}")
            
            # Replace documents with only valid ones
            documents = valid_documents
            
            # Log content validation summary
            logger.info(f"Content validation complete: {len(documents)} valid documents, {total_content_size} total chars")
            if content_validation:
                logger.info(f"Content hashes for verification: {json.dumps(content_validation, indent=2)}")
            
            # Enhanced content validation - check if we got real RFP content
            if not self._validate_rfp_content(documents, total_content_size):
                logger.warning(f"Content validation failed ({total_content_size} chars), attempting fallback scraping")
                page_content = await self._extract_page_content_fallback(url)
                if page_content and len(page_content) > 1000:
                    content_hash = hashlib.sha256(page_content.encode()).hexdigest()[:16]
                    documents['fallback_content.txt'] = page_content
                    total_content_size += len(page_content)
                    logger.info(f"Added fallback content: {len(page_content)} chars, hash: {content_hash}")
            
            # Extract and download attachments from URLs in the content
            attachment_urls = await self._extract_attachment_urls(documents, url)
            if attachment_urls:
                logger.info(f"Found {len(attachment_urls)} attachment URLs to download")
                downloaded_attachments = await self._download_attachments(attachment_urls, notice_id)
                documents.update(downloaded_attachments)
                logger.info(f"Downloaded {len(downloaded_attachments)} attachments")
            
            # Store content validation in metadata for downstream verification
            rfp_metadata['content_validation'] = content_validation
            rfp_metadata['extraction_stats'] = {
                'total_documents': len(documents),
                'total_size': total_content_size,
                'extraction_method': 'browser-use' if len(documents) > 1 else 'fallback',
                'attachment_urls': attachment_urls
            }
            
            logger.info(f"Successfully scraped {len(documents)} documents for {notice_id} with {total_content_size} total chars")
            return documents, rfp_metadata
            
        except Exception as e:
            logger.error(f"Error with Browser Use for RFP {notice_id}: {e}")
            # Fallback to simple HTTP scraping
            return await self._fallback_scraping(url, notice_id)
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            reader = pypdf.PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return f"Error extracting PDF content: {e}"
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            import docx
            doc = docx.Document(str(file_path))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return f"Error extracting DOCX content: {e}"
    
    async def _extract_page_content_fallback(self, url: str) -> str:
        """Extract main content from webpage using simple HTTP request"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        return text
        except Exception as e:
            logger.error(f"Error extracting page content: {e}")
            return f"Error extracting page content: {e}"
    
    async def _fallback_scraping(self, url: str, notice_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fallback scraping method using direct HTTP requests"""
        try:
            logger.info(f"Using fallback HTTP scraping for {notice_id}")
            
            content = await self._extract_page_content_fallback(url)
            
            documents = {
                'webpage_content.txt': content
            }
            
            metadata = {
                'notice_id': notice_id,
                'url': url,
                'method': 'fallback_http',
                'content_length': len(content)
            }
            
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Fallback scraping failed: {e}")
            return {}, {'notice_id': notice_id, 'error': str(e)}
    
    def _validate_rfp_content(self, documents: Dict, total_size: int) -> bool:
        """Validate that we have actual RFP content, not just action logs"""
        if not documents or total_size < 1000:
            return False
        
        # Check for RFP-related keywords
        required_keywords = [
            'solicitation', 'rfp', 'rfi', 'rfq', 'proposal',
            'due date', 'deadline', 'submission', 'response',
            'contract', 'award', 'evaluation', 'requirements'
        ]
        
        all_text = ' '.join(str(doc) for doc in documents.values()).lower()
        found_keywords = sum(1 for kw in required_keywords if kw in all_text)
        
        # Need at least 3 RFP keywords and 5000 chars to be valid
        return found_keywords >= 3 and total_size >= 5000
    
    async def _extract_attachment_urls(self, documents: Dict[str, str], base_url: str) -> List[str]:
        """Extract attachment URLs from the scraped content"""
        attachment_urls = []
        
        # Combine all document content for URL extraction
        all_content = ' '.join(str(doc) for doc in documents.values())
        
        # Look for ATTACHMENT_URLS section that we asked Browser Use to create
        if 'ATTACHMENT_URLS:' in all_content:
            urls_section = all_content.split('ATTACHMENT_URLS:')[1]
            # Extract URLs from this section
            url_patterns = [
                r'https?://[^\s<>"]+\.(?:pdf|docx?|xlsx?|zip)',
                r'/api/opportunities/v2/download/[^\s<>"]+',
                r'/api/prod/opps/v3/opportunities/[^/]+/resources/download/[^\s<>"]+'
            ]
            for pattern in url_patterns:
                matches = re.findall(pattern, urls_section, re.IGNORECASE)
                attachment_urls.extend(matches)
        
        # Also search entire content for attachment URLs
        url_patterns = [
            # SAM.gov download API patterns
            r'https://sam\.gov/api/prod/opps/v3/opportunities/[^/]+/resources/download/[^\s<>"]+',
            r'/api/prod/opps/v3/opportunities/[^/]+/resources/download/[^\s<>"]+',
            r'https://sam\.gov/api/opportunities/v2/download/[^\s<>"]+',
            r'/api/opportunities/v2/download/[^\s<>"]+',
            # Direct file links
            r'https?://[^\s<>"]+\.pdf(?:\?[^\s<>"]*)?',
            r'https?://[^\s<>"]+\.docx?(?:\?[^\s<>"]*)?',
            r'https?://[^\s<>"]+\.xlsx?(?:\?[^\s<>"]*)?'
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            for match in matches:
                # Convert relative URLs to absolute
                if match.startswith('/'):
                    if 'sam.gov' in base_url:
                        match = 'https://sam.gov' + match
                    else:
                        from urllib.parse import urlparse
                        parsed = urlparse(base_url)
                        match = f"{parsed.scheme}://{parsed.netloc}{match}"
                
                if match not in attachment_urls:
                    attachment_urls.append(match)
                    logger.info(f"Found attachment URL: {match}")
        
        return attachment_urls
    
    async def _download_attachments(self, urls: List[str], notice_id: str) -> Dict[str, str]:
        """Download attachments from URLs"""
        downloaded = {}
        
        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    # Extract filename from URL
                    filename = url.split('/')[-1].split('?')[0]
                    if not filename or '.' not in filename:
                        filename = f"attachment_{len(downloaded) + 1}.pdf"
                    
                    logger.info(f"Downloading attachment: {filename} from {url}")
                    
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            # Save to temp directory
                            file_path = self.download_dir / filename
                            file_path.write_bytes(content)
                            
                            # Extract text content based on file type
                            if filename.endswith('.pdf'):
                                text_content = self._extract_pdf_text(file_path)
                            elif filename.endswith(('.docx', '.doc')):
                                text_content = self._extract_docx_text(file_path)
                            elif filename.endswith(('.txt', '.md')):
                                text_content = file_path.read_text(encoding='utf-8', errors='ignore')
                            else:
                                text_content = f"Downloaded binary file: {filename} ({len(content)} bytes)"
                            
                            downloaded[f"attachment_{filename}"] = text_content
                            logger.info(f"Successfully downloaded and extracted {filename}: {len(text_content)} chars")
                        else:
                            logger.warning(f"Failed to download {url}: HTTP {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error downloading {url}: {e}")
                    continue
        
        return downloaded
    
    def cleanup(self):
        """Clean up temporary download directory"""
        try:
            import shutil
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up downloads: {e}")