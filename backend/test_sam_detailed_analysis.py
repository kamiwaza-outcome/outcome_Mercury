#!/usr/bin/env python3
"""
SAM.gov API Detailed Analysis
Examines what data is actually returned and tries different date ranges
"""
import aiohttp
import asyncio
import json
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMDetailedTester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.test_notice_id = '710637f47916457280ca93dbb73c2b41'
        self.headers = {"Accept": "application/json"}
    
    async def examine_api_response_structure(self):
        """Examine what the API actually returns"""
        logger.info("=== EXAMINING API RESPONSE STRUCTURE ===")
        
        # Try a basic search to see what comes back
        url = 'https://api.sam.gov/prod/opportunities/v2/search'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        params = {
            'api_key': self.api_key,
            'limit': 3,
            'postedFrom': start_date.strftime("%m/%d/%Y"),
            'postedTo': end_date.strftime("%m/%d/%Y")
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        logger.info(f"Response keys: {list(data.keys())}")
                        
                        opportunities = data.get('opportunitiesData', [])
                        logger.info(f"Found {len(opportunities)} opportunities")
                        
                        if opportunities:
                            logger.info("\n--- SAMPLE OPPORTUNITY STRUCTURE ---")
                            sample = opportunities[0]
                            logger.info(f"Sample opportunity keys: {list(sample.keys())}")
                            
                            # Print key fields
                            for key in ['noticeId', 'solicitationNumber', 'title', 'postedDate', 'type']:
                                if key in sample:
                                    logger.info(f"{key}: {sample[key]}")
                            
                            # Print all notice IDs found
                            logger.info("\n--- ALL NOTICE IDs IN RESPONSE ---")
                            for i, opp in enumerate(opportunities):
                                notice_id = opp.get('noticeId', 'N/A')
                                title = opp.get('title', 'N/A')[:60] + '...' if len(opp.get('title', '')) > 60 else opp.get('title', 'N/A')
                                logger.info(f"{i+1}. {notice_id} - {title}")
                        
                        return data
                    else:
                        logger.error(f"API error: {response.status}")
                        error_text = await response.text()
                        logger.error(f"Error: {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
    
    async def test_different_date_ranges(self):
        """Test various date ranges to see if the notice exists in older data"""
        logger.info("\n=== TESTING DIFFERENT DATE RANGES ===")
        
        url = 'https://api.sam.gov/prod/opportunities/v2/search'
        
        # Define different date ranges to test
        now = datetime.now()
        date_ranges = [
            ("Last 30 days", now - timedelta(days=30), now),
            ("Last 90 days", now - timedelta(days=90), now),
            ("Last 180 days", now - timedelta(days=180), now),
            ("2024 Q4", datetime(2024, 10, 1), datetime(2024, 12, 31)),
            ("2024 Q3", datetime(2024, 7, 1), datetime(2024, 9, 30)),
            ("2024 Q2", datetime(2024, 4, 1), datetime(2024, 6, 30)),
            ("2024 Q1", datetime(2024, 1, 1), datetime(2024, 3, 31)),
            ("2023 Q4", datetime(2023, 10, 1), datetime(2023, 12, 31)),
        ]
        
        for range_name, start_date, end_date in date_ranges:
            logger.info(f"\n--- Testing {range_name} ({start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}) ---")
            
            params = {
                'api_key': self.api_key,
                'noticeId': self.test_notice_id,
                'limit': 10,
                'postedFrom': start_date.strftime("%m/%d/%Y"),
                'postedTo': end_date.strftime("%m/%d/%Y")
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            opportunities = data.get('opportunitiesData', [])
                            
                            logger.info(f"Status: {response.status}, Opportunities: {len(opportunities)}")
                            
                            # Check if our target notice is found
                            found_target = False
                            for opp in opportunities:
                                if opp.get('noticeId') == self.test_notice_id:
                                    found_target = True
                                    logger.info(f"✅ FOUND TARGET NOTICE! Title: {opp.get('title')}")
                                    logger.info(f"Posted Date: {opp.get('postedDate')}")
                                    break
                            
                            if not found_target and opportunities:
                                logger.info("❌ Target not found, but found other notices:")
                                for opp in opportunities[:3]:  # Show first 3
                                    logger.info(f"  - {opp.get('noticeId')} - {opp.get('title', 'No title')[:50]}...")
                            elif not found_target:
                                logger.info("❌ No opportunities found in this date range")
                        else:
                            logger.error(f"API error: {response.status}")
                            error_text = await response.text()
                            logger.error(f"Error: {error_text[:200]}...")
                            
            except Exception as e:
                logger.error(f"Error testing {range_name}: {e}")
            
            await asyncio.sleep(1)  # Rate limiting
    
    async def test_search_without_notice_id(self):
        """Search without notice ID to see if we can find it in general results"""
        logger.info("\n=== SEARCHING WITHOUT NOTICE ID ===")
        
        url = 'https://api.sam.gov/prod/opportunities/v2/search'
        
        # Search in different time periods with larger limits
        time_periods = [
            ("Recent 90 days", datetime.now() - timedelta(days=90), datetime.now(), 100),
            ("2024 Full Year", datetime(2024, 1, 1), datetime(2024, 12, 31), 100),
            ("2023 Full Year", datetime(2023, 1, 1), datetime(2023, 12, 31), 100),
        ]
        
        for period_name, start_date, end_date, limit in time_periods:
            logger.info(f"\n--- Searching {period_name} ---")
            
            params = {
                'api_key': self.api_key,
                'limit': limit,
                'postedFrom': start_date.strftime("%m/%d/%Y"),
                'postedTo': end_date.strftime("%m/%d/%Y")
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            opportunities = data.get('opportunitiesData', [])
                            
                            logger.info(f"Found {len(opportunities)} opportunities")
                            
                            # Look for our target notice ID
                            found_target = False
                            for opp in opportunities:
                                if opp.get('noticeId') == self.test_notice_id:
                                    found_target = True
                                    logger.info(f"✅ FOUND TARGET NOTICE IN GENERAL SEARCH!")
                                    logger.info(f"Title: {opp.get('title')}")
                                    logger.info(f"Posted Date: {opp.get('postedDate')}")
                                    logger.info(f"Type: {opp.get('type')}")
                                    break
                            
                            if not found_target:
                                logger.info(f"❌ Target notice {self.test_notice_id} not found in {len(opportunities)} opportunities")
                                # Show some sample notice IDs
                                logger.info("Sample notice IDs found:")
                                for i, opp in enumerate(opportunities[:5]):
                                    logger.info(f"  {i+1}. {opp.get('noticeId')}")
                        else:
                            logger.error(f"API error: {response.status}")
                            
            except Exception as e:
                logger.error(f"Error in general search: {e}")
            
            await asyncio.sleep(2)  # Longer pause for large searches
    
    async def test_alternative_api_endpoints(self):
        """Test alternative SAM.gov API endpoints"""
        logger.info("\n=== TESTING ALTERNATIVE ENDPOINTS ===")
        
        endpoints_to_test = [
            # Different versions
            ("v2 without /prod/", "https://api.sam.gov/opportunities/v2/search"),
            ("v1 without /prod/", "https://api.sam.gov/opportunities/v1/search"), 
            # Entity API (for vendor lookups)
            ("Entity API test", "https://api.sam.gov/prod/entity/v3/search"),
        ]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        for endpoint_name, url in endpoints_to_test:
            logger.info(f"\n--- Testing {endpoint_name} ---")
            logger.info(f"URL: {url}")
            
            if "entity" in url:
                # Different params for entity API
                params = {
                    'api_key': self.api_key,
                    'limit': 5,
                    'entityName': 'test'
                }
            else:
                params = {
                    'api_key': self.api_key,
                    'noticeId': self.test_notice_id,
                    'limit': 10,
                    'postedFrom': start_date.strftime("%m/%d/%Y"),
                    'postedTo': end_date.strftime("%m/%d/%Y")
                }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                        status = response.status
                        response_text = await response.text()
                        
                        logger.info(f"Status: {status}")
                        
                        if status == 200:
                            try:
                                data = json.loads(response_text)
                                if isinstance(data, dict):
                                    logger.info(f"Response keys: {list(data.keys())}")
                                    
                                    if 'opportunitiesData' in data:
                                        opps = data['opportunitiesData']
                                        logger.info(f"Found {len(opps)} opportunities")
                                        
                                        # Check for target
                                        for opp in opps:
                                            if opp.get('noticeId') == self.test_notice_id:
                                                logger.info("✅ FOUND TARGET NOTICE!")
                                                return
                                    elif 'entityData' in data:
                                        logger.info(f"Entity API working - found {len(data.get('entityData', []))} entities")
                                else:
                                    logger.info(f"Response type: {type(data)}")
                            except:
                                logger.info("Response not JSON")
                        else:
                            logger.error(f"Error response: {response_text[:200]}...")
                            
            except Exception as e:
                logger.error(f"Error testing {endpoint_name}: {e}")
            
            await asyncio.sleep(1)
    
    async def run_detailed_analysis(self):
        """Run all detailed analysis tests"""
        logger.info("Starting detailed SAM.gov API analysis...")
        
        await self.examine_api_response_structure()
        await self.test_different_date_ranges()  
        await self.test_search_without_notice_id()
        await self.test_alternative_api_endpoints()
        
        logger.info("\n=== ANALYSIS COMPLETE ===")

async def main():
    api_key = "KXHzDyofJ8WQXz5JuPt4Y3oNiYmFOx3803lqnbrs"
    tester = SAMDetailedTester(api_key)
    await tester.run_detailed_analysis()

if __name__ == "__main__":
    asyncio.run(main())