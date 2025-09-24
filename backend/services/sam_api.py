import os
import logging
import aiohttp
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class SamApiClient:
    def __init__(self):
        self.api_key = os.getenv("SAM_API_KEY")
        self.base_url = "https://api.sam.gov/prod/opportunities/v2"
        # Note: SAM.gov API requires the key as a query parameter, not a header
        self.headers = {
            "Accept": "application/json"
        }
        
        # Validate API key
        if not self.api_key:
            logger.warning("SAM_API_KEY environment variable not set. API calls will fail.")
        elif len(self.api_key) != 40:
            logger.warning(f"SAM_API_KEY appears invalid (expected 40 chars, got {len(self.api_key)}). API calls may fail.")
    
    async def get_opportunity(self, notice_id: str) -> Dict[str, Any]:
        """Get opportunity details from SAM.gov API"""
        if not self.api_key:
            logger.error("Cannot fetch opportunity: SAM_API_KEY not configured")
            return {}
            
        try:
            from datetime import timedelta
            url = f"{self.base_url}/search"
            # Use last 90 days to stay within reasonable range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            params = {
                "api_key": self.api_key,  # API key must be in query params
                "noticeId": notice_id,
                "limit": 1,
                "postedFrom": start_date.strftime("%m/%d/%Y"),
                "postedTo": end_date.strftime("%m/%d/%Y")
            }
            
            logger.info(f"SAM API request: {url} with params: {params}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        opportunities = data.get('opportunitiesData', [])
                        logger.info(f"SAM API returned {len(opportunities)} opportunities for notice_id {notice_id}")
                        
                        if opportunities:
                            # CRITICAL: Validate that the returned opportunity matches our requested notice_id
                            for opp in opportunities:
                                if opp.get('noticeId') == notice_id:
                                    logger.info(f"SAM API found matching opportunity - notice_id: {opp.get('noticeId')} title: {opp.get('title')}")
                                    return self._parse_opportunity(opp)

                            # Log what we got instead
                            first_opp = opportunities[0] if opportunities else {}
                            logger.error(f"SAM API returned wrong opportunity! Requested: {notice_id}, Got: {first_opp.get('noticeId')} - {first_opp.get('title')}")
                            logger.warning(f"No matching opportunity found for notice ID: {notice_id}")
                            return {}
                        else:
                            logger.warning(f"No opportunity found for notice ID: {notice_id}")
                            return {}
                    else:
                        logger.error(f"SAM API error: {response.status}")
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")
                        
                        # Provide helpful error messages
                        if response.status == 403:
                            logger.error("403 Forbidden: Check if SAM_API_KEY is valid and has proper permissions")
                        elif response.status == 401:
                            logger.error("401 Unauthorized: SAM_API_KEY may be missing or invalid")
                        elif response.status == 429:
                            logger.error("429 Too Many Requests: Rate limit exceeded")
                        
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching opportunity {notice_id}: {e}")
            return {}
    
    def _parse_opportunity(self, opp_data: Dict) -> Dict[str, Any]:
        """Parse opportunity data into structured format"""
        return {
            'notice_id': opp_data.get('noticeId'),
            'title': opp_data.get('title'),
            'sol_number': opp_data.get('solicitationNumber'),
            'department': opp_data.get('department'),
            'agency': opp_data.get('agency'),
            'description': opp_data.get('description'),
            'type': opp_data.get('type'),
            'naics': opp_data.get('naicsCode'),
            'classification_code': opp_data.get('classificationCode'),
            'posted_date': opp_data.get('postedDate'),
            'response_deadline': opp_data.get('responseDeadline'),
            'archive_date': opp_data.get('archiveDate'),
            'point_of_contact': self._parse_contact(opp_data.get('pointOfContact', [])),
            'place_of_performance': opp_data.get('placeOfPerformance'),
            'additional_info_link': opp_data.get('additionalInfoLink'),
            'ui_link': opp_data.get('uiLink'),
            'resource_links': opp_data.get('resourceLinks', []),
            'award': opp_data.get('award')
        }
    
    def _parse_contact(self, contacts: List[Dict]) -> List[Dict]:
        """Parse point of contact information"""
        parsed_contacts = []
        for contact in contacts:
            if isinstance(contact, dict):
                parsed_contacts.append({
                    'name': contact.get('fullName'),
                    'title': contact.get('title'),
                    'email': contact.get('email'),
                    'phone': contact.get('phone'),
                    'fax': contact.get('fax')
                })
        return parsed_contacts
    
    async def download_attachments(self, notice_id: str) -> Dict[str, str]:
        """Download attachments for an opportunity"""
        documents = {}
        
        if not self.api_key:
            logger.warning("Cannot download attachments: SAM_API_KEY not configured")
            return documents
        
        try:
            opp_data = await self.get_opportunity(notice_id)
            
            if not opp_data:
                logger.warning(f"No opportunity data found for {notice_id}")
                return documents
            
            resource_links = opp_data.get('resource_links', [])
            
            for idx, link in enumerate(resource_links):
                if isinstance(link, str):
                    doc_name = f"attachment_{idx}.pdf"
                    content = await self._download_file(link)
                    if content:
                        documents[doc_name] = content
                elif isinstance(link, dict):
                    doc_name = link.get('name', f"attachment_{idx}")
                    url = link.get('url', link.get('link'))
                    if url:
                        content = await self._download_file(url)
                        if content:
                            documents[doc_name] = content
            
            additional_link = opp_data.get('additional_info_link')
            if additional_link:
                content = await self._download_file(additional_link)
                if content:
                    documents['additional_info.pdf'] = content
            
            if opp_data.get('description'):
                documents['description.txt'] = opp_data['description']
            
            logger.info(f"Downloaded {len(documents)} documents for {notice_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Error downloading attachments for {notice_id}: {e}")
            return documents
    
    async def _download_file(self, url: str) -> Optional[str]:
        """Download a file from URL"""
        try:
            # Add API key to URL if it's a SAM.gov URL
            if 'sam.gov' in url and self.api_key and '?' in url:
                url = f"{url}&api_key={self.api_key}"
            elif 'sam.gov' in url and self.api_key:
                url = f"{url}?api_key={self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=60) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        try:
                            text_content = content.decode('utf-8')
                            return text_content
                        except UnicodeDecodeError:
                            return f"[Binary file downloaded from {url}]"
                    else:
                        logger.warning(f"Failed to download file from {url}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {e}")
            return None
    
    async def search_opportunities(self, keywords: str = None, posted_from: str = None, posted_to: str = None) -> List[Dict]:
        """Search for opportunities"""
        if not self.api_key:
            logger.error("Cannot search opportunities: SAM_API_KEY not configured")
            return []
            
        try:
            url = f"{self.base_url}/search"
            params = {
                "api_key": self.api_key,  # API key must be in query params
                "limit": 100,
                "postedFrom": posted_from or "01/01/2024",
                "postedTo": posted_to or datetime.now().strftime("%m/%d/%Y")
            }
            
            if keywords:
                params["keywords"] = keywords
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        opportunities = data.get('opportunitiesData', [])
                        return [self._parse_opportunity(opp) for opp in opportunities]
                    else:
                        logger.error(f"SAM API search error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error searching opportunities: {e}")
            return []