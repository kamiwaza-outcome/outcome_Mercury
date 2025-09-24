#!/usr/bin/env python3
"""
SAM.gov API Test Script
Tests different parameter variations to identify what works vs what doesn't
"""
import aiohttp
import asyncio
import json
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMApiTester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.test_notice_id = '710637f47916457280ca93dbb73c2b41'
        self.headers = {"Accept": "application/json"}
        self.results = []
    
    async def test_variation(self, test_name: str, url: str, params: dict):
        """Test a specific API variation"""
        logger.info(f"\n=== Testing: {test_name} ===")
        logger.info(f"URL: {url}")
        logger.info(f"Params: {json.dumps(params, indent=2)}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                    status = response.status
                    response_text = await response.text()
                    
                    # Try to parse as JSON
                    try:
                        response_json = json.loads(response_text)
                    except:
                        response_json = None
                    
                    # Analyze response
                    opportunities_count = 0
                    found_target = False
                    if response_json and isinstance(response_json, dict):
                        opportunities = response_json.get('opportunitiesData', [])
                        opportunities_count = len(opportunities)
                        
                        # Check if we found our target notice
                        for opp in opportunities:
                            if opp.get('noticeId') == self.test_notice_id:
                                found_target = True
                                break
                    
                    result = {
                        'test_name': test_name,
                        'url': url,
                        'params': params,
                        'status_code': status,
                        'opportunities_count': opportunities_count,
                        'found_target_notice': found_target,
                        'response_size': len(response_text),
                        'success': status == 200 and found_target
                    }
                    
                    # Store full response for successful tests
                    if status == 200:
                        result['response_preview'] = response_text[:500] + "..." if len(response_text) > 500 else response_text
                        if response_json:
                            result['response_structure'] = list(response_json.keys()) if isinstance(response_json, dict) else type(response_json).__name__
                    else:
                        result['error_response'] = response_text[:500]
                    
                    self.results.append(result)
                    
                    logger.info(f"Status: {status}")
                    logger.info(f"Opportunities found: {opportunities_count}")
                    logger.info(f"Target notice found: {found_target}")
                    logger.info(f"SUCCESS: {result['success']}")
                    
                    return result
                    
        except Exception as e:
            logger.error(f"Error in test '{test_name}': {e}")
            result = {
                'test_name': test_name,
                'url': url,
                'params': params,
                'error': str(e),
                'success': False
            }
            self.results.append(result)
            return result
    
    async def run_all_tests(self):
        """Run all API variation tests"""
        logger.info("Starting SAM.gov API variation tests...")
        
        # Date range setup
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        date_from = start_date.strftime("%m/%d/%Y")
        date_to = end_date.strftime("%m/%d/%Y")
        
        # Test variations
        tests = [
            # Test 1: Original approach with /prod/ URL and noticeId (camelCase)
            {
                'name': 'Original: /prod/ + noticeId (camelCase)',
                'url': 'https://api.sam.gov/prod/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeId': self.test_notice_id,
                    'limit': 10,
                    'postedFrom': date_from,
                    'postedTo': date_to
                }
            },
            
            # Test 2: Try with noticeid (all lowercase)
            {
                'name': '/prod/ + noticeid (lowercase)',
                'url': 'https://api.sam.gov/prod/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeid': self.test_notice_id,
                    'limit': 10,
                    'postedFrom': date_from,
                    'postedTo': date_to
                }
            },
            
            # Test 3: Try without /prod/ in URL with noticeId
            {
                'name': 'Without /prod/ + noticeId (camelCase)',
                'url': 'https://api.sam.gov/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeId': self.test_notice_id,
                    'limit': 10,
                    'postedFrom': date_from,
                    'postedTo': date_to
                }
            },
            
            # Test 4: Try without /prod/ in URL with noticeid
            {
                'name': 'Without /prod/ + noticeid (lowercase)',
                'url': 'https://api.sam.gov/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeid': self.test_notice_id,
                    'limit': 10,
                    'postedFrom': date_from,
                    'postedTo': date_to
                }
            },
            
            # Test 5: Without date range + noticeId
            {
                'name': '/prod/ + noticeId - NO DATE RANGE',
                'url': 'https://api.sam.gov/prod/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeId': self.test_notice_id,
                    'limit': 10
                }
            },
            
            # Test 6: Without date range + noticeid
            {
                'name': '/prod/ + noticeid - NO DATE RANGE',
                'url': 'https://api.sam.gov/prod/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeid': self.test_notice_id,
                    'limit': 10
                }
            },
            
            # Test 7: Try with different date range (last 365 days)
            {
                'name': '/prod/ + noticeId - EXTENDED DATE RANGE (365 days)',
                'url': 'https://api.sam.gov/prod/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeId': self.test_notice_id,
                    'limit': 10,
                    'postedFrom': (datetime.now() - timedelta(days=365)).strftime("%m/%d/%Y"),
                    'postedTo': date_to
                }
            },
            
            # Test 8: Try exact ID lookup (if API supports direct ID endpoint)
            {
                'name': 'Direct ID lookup attempt',
                'url': f'https://api.sam.gov/prod/opportunities/v2/{self.test_notice_id}',
                'params': {
                    'api_key': self.api_key
                }
            },
            
            # Test 9: General search without notice ID to see if API works at all
            {
                'name': 'General search - NO NOTICE ID (test API connectivity)',
                'url': 'https://api.sam.gov/prod/opportunities/v2/search',
                'params': {
                    'api_key': self.api_key,
                    'limit': 5,
                    'postedFrom': date_from,
                    'postedTo': date_to
                }
            },
            
            # Test 10: Try v1 API
            {
                'name': 'v1 API + noticeId',
                'url': 'https://api.sam.gov/prod/opportunities/v1/search',
                'params': {
                    'api_key': self.api_key,
                    'noticeId': self.test_notice_id,
                    'limit': 10,
                    'postedFrom': date_from,
                    'postedTo': date_to
                }
            }
        ]
        
        # Run all tests
        for test in tests:
            await self.test_variation(test['name'], test['url'], test['params'])
            await asyncio.sleep(1)  # Be nice to the API
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "="*80)
        print("SAM.gov API TEST RESULTS SUMMARY")
        print("="*80)
        
        successful_tests = [r for r in self.results if r.get('success', False)]
        failed_tests = [r for r in self.results if not r.get('success', False)]
        
        print(f"\nSUCCESSFUL TESTS ({len(successful_tests)}):")
        print("-" * 50)
        for result in successful_tests:
            print(f"✅ {result['test_name']}")
            print(f"   URL: {result['url']}")
            print(f"   Found {result['opportunities_count']} opportunities")
            print(f"   Target notice found: {result['found_target_notice']}")
            if 'response_structure' in result:
                print(f"   Response structure: {result['response_structure']}")
            print()
        
        print(f"\nFAILED TESTS ({len(failed_tests)}):")
        print("-" * 50)
        for result in failed_tests:
            print(f"❌ {result['test_name']}")
            print(f"   URL: {result['url']}")
            if 'status_code' in result:
                print(f"   Status: {result['status_code']}")
                if result.get('opportunities_count', 0) > 0:
                    print(f"   Found {result['opportunities_count']} opportunities (but not target)")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            if 'error_response' in result:
                print(f"   Response: {result['error_response']}")
            print()
        
        # Key findings
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        
        if successful_tests:
            print(f"✅ {len(successful_tests)} test(s) successfully found the target notice ID")
            for result in successful_tests:
                print(f"   - {result['test_name']}")
        else:
            print("❌ NO tests successfully found the target notice ID")
        
        # Analyze patterns
        param_patterns = {}
        url_patterns = {}
        
        for result in self.results:
            if result.get('success'):
                # Analyze URL patterns
                url = result['url']
                if '/prod/' in url:
                    url_patterns['with_prod'] = url_patterns.get('with_prod', 0) + 1
                else:
                    url_patterns['without_prod'] = url_patterns.get('without_prod', 0) + 1
                
                # Analyze param patterns
                params = result['params']
                if 'noticeId' in params:
                    param_patterns['camelCase'] = param_patterns.get('camelCase', 0) + 1
                if 'noticeid' in params:
                    param_patterns['lowercase'] = param_patterns.get('lowercase', 0) + 1
                if 'postedFrom' in params:
                    param_patterns['with_dates'] = param_patterns.get('with_dates', 0) + 1
                else:
                    param_patterns['without_dates'] = param_patterns.get('without_dates', 0) + 1
        
        if url_patterns:
            print(f"\nURL patterns in successful tests: {url_patterns}")
        if param_patterns:
            print(f"Parameter patterns in successful tests: {param_patterns}")
        
        print("\n" + "="*80)

async def main():
    api_key = "KXHzDyofJ8WQXz5JuPt4Y3oNiYmFOx3803lqnbrs"
    tester = SAMApiTester(api_key)
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())