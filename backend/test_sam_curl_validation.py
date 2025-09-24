#!/usr/bin/env python3
"""
SAM.gov API Curl Validation Test
Uses curl and tests with real notice IDs found in the API to validate our approach
"""
import subprocess
import json
import sys
from datetime import datetime, timedelta

def run_curl_command(url, params):
    """Run curl command and return results"""
    # Build curl command
    cmd = ["curl", "-s", "-G", url]
    
    # Add parameters
    for key, value in params.items():
        cmd.extend(["-d", f"{key}={value}"])
    
    cmd.extend(["-H", "Accept: application/json"])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout after 30 seconds"
    except Exception as e:
        return -1, "", str(e)

def test_sam_api_with_curl():
    """Test SAM API using curl with different variations"""
    
    api_key = "KXHzDyofJ8WQXz5JuPt4Y3oNiYmFOx3803lqnbrs"
    target_notice_id = "710637f47916457280ca93dbb73c2b41"
    
    # Real notice IDs we found in the API (from previous test)
    real_notice_ids = [
        "fff1536501f94368b2de1421cef00d8f",  # Recent microphone
        "ffd3cf65e0be4cfdbab6329e38f2a92b",  # Recent valve seat
        "df3f56a8fe5d4f1683b9a786fb7037e1",  # 2024 Q4 chromium
        "d3b40d4c034f49b481e01442fc18ab3b",  # 2024 Q2 schedule
    ]
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_from = start_date.strftime("%m/%d/%Y")
    date_to = end_date.strftime("%m/%d/%Y")
    
    print("="*80)
    print("SAM.gov API CURL VALIDATION TESTS")
    print("="*80)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Target Notice ID - noticeId (camelCase)',
            'url': 'https://api.sam.gov/prod/opportunities/v2/search',
            'params': {
                'api_key': api_key,
                'noticeId': target_notice_id,
                'limit': '10',
                'postedFrom': date_from,
                'postedTo': date_to
            }
        },
        {
            'name': 'Target Notice ID - noticeid (lowercase)',  
            'url': 'https://api.sam.gov/prod/opportunities/v2/search',
            'params': {
                'api_key': api_key,
                'noticeid': target_notice_id,
                'limit': '10',
                'postedFrom': date_from,
                'postedTo': date_to
            }
        },
        {
            'name': 'KNOWN REAL Notice ID - noticeId (camelCase)',
            'url': 'https://api.sam.gov/prod/opportunities/v2/search',
            'params': {
                'api_key': api_key,
                'noticeId': real_notice_ids[0],  # Use first real ID
                'limit': '10',
                'postedFrom': date_from,
                'postedTo': date_to
            }
        },
        {
            'name': 'KNOWN REAL Notice ID - noticeid (lowercase)',
            'url': 'https://api.sam.gov/prod/opportunities/v2/search', 
            'params': {
                'api_key': api_key,
                'noticeid': real_notice_ids[0],  # Use first real ID
                'limit': '10',
                'postedFrom': date_from,
                'postedTo': date_to
            }
        },
        {
            'name': 'Without /prod/ - Target Notice ID',
            'url': 'https://api.sam.gov/opportunities/v2/search',
            'params': {
                'api_key': api_key,
                'noticeId': target_notice_id,
                'limit': '10',
                'postedFrom': date_from,
                'postedTo': date_to
            }
        },
        {
            'name': 'General Search - No Notice ID (baseline test)',
            'url': 'https://api.sam.gov/prod/opportunities/v2/search',
            'params': {
                'api_key': api_key,
                'limit': '5',
                'postedFrom': date_from,
                'postedTo': date_to
            }
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n{'-'*60}")
        print(f"TEST: {test_config['name']}")
        print(f"{'-'*60}")
        
        returncode, stdout, stderr = run_curl_command(test_config['url'], test_config['params'])
        
        result = {
            'name': test_config['name'],
            'url': test_config['url'],
            'params': test_config['params'],
            'returncode': returncode,
            'success': False,
            'found_target': False,
            'found_real_id': False,
            'opportunities_count': 0
        }
        
        if returncode == 0:
            try:
                # Parse JSON response
                data = json.loads(stdout)
                
                if isinstance(data, dict) and 'opportunitiesData' in data:
                    opportunities = data['opportunitiesData']
                    result['opportunities_count'] = len(opportunities)
                    result['success'] = True
                    
                    print(f"✅ SUCCESS: Found {len(opportunities)} opportunities")
                    
                    # Check for target notice ID
                    for opp in opportunities:
                        notice_id = opp.get('noticeId', '')
                        if notice_id == target_notice_id:
                            result['found_target'] = True
                            print(f"🎯 FOUND TARGET NOTICE: {notice_id}")
                            print(f"   Title: {opp.get('title', 'N/A')}")
                        
                        # Check if it's one of our known real IDs
                        if notice_id in real_notice_ids:
                            result['found_real_id'] = True
                            print(f"✅ Found known real ID: {notice_id}")
                    
                    if not result['found_target'] and not result['found_real_id']:
                        print("ℹ️  Notice IDs in response:")
                        for opp in opportunities[:3]:  # Show first 3
                            print(f"   - {opp.get('noticeId', 'N/A')}")
                
                elif isinstance(data, dict) and 'errorCode' in data:
                    print(f"❌ API ERROR: {data.get('errorCode')} - {data.get('errorMessage', 'Unknown error')}")
                    result['error'] = f"{data.get('errorCode')}: {data.get('errorMessage')}"
                
                else:
                    print(f"❌ UNEXPECTED RESPONSE FORMAT: {type(data)}")
                    print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
            except json.JSONDecodeError:
                print(f"❌ INVALID JSON RESPONSE")
                print(f"Response: {stdout[:200]}...")
                result['error'] = "Invalid JSON response"
        else:
            print(f"❌ CURL FAILED (exit code {returncode})")
            if stderr:
                print(f"Error: {stderr}")
            result['error'] = f"Curl failed: {stderr}"
        
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    
    successful_tests = [r for r in results if r['success']]
    target_found_tests = [r for r in results if r['found_target']]
    real_id_tests = [r for r in results if r['found_real_id']]
    
    print(f"\nSuccessful API calls: {len(successful_tests)}/{len(results)}")
    print(f"Tests that found target notice ID: {len(target_found_tests)}")
    print(f"Tests that found known real notice IDs: {len(real_id_tests)}")
    
    if target_found_tests:
        print(f"\n🎯 TARGET NOTICE ID FOUND BY:")
        for result in target_found_tests:
            print(f"   ✅ {result['name']}")
    else:
        print(f"\n❌ TARGET NOTICE ID '{target_notice_id}' NOT FOUND BY ANY TEST")
    
    # Pattern analysis
    print(f"\n📊 PATTERN ANALYSIS:")
    
    # Check parameter casing patterns
    camel_case_success = [r for r in successful_tests if 'noticeId' in r['params']]
    lowercase_success = [r for r in successful_tests if 'noticeid' in r['params']]
    
    print(f"   - Tests with 'noticeId' (camelCase): {len(camel_case_success)} successful")
    print(f"   - Tests with 'noticeid' (lowercase): {len(lowercase_success)} successful")
    
    # Check URL patterns
    prod_url_success = [r for r in successful_tests if '/prod/' in r['url']]
    no_prod_success = [r for r in successful_tests if '/prod/' not in r['url']]
    
    print(f"   - Tests with /prod/ in URL: {len(prod_url_success)} successful")
    print(f"   - Tests without /prod/ in URL: {len(no_prod_success)} successful")
    
    # Key findings
    print(f"\n🔑 KEY FINDINGS:")
    if successful_tests:
        print(f"   ✅ SAM.gov API is accessible and working")
        print(f"   ✅ API key is valid and functional")
        
        if not target_found_tests:
            print(f"   ⚠️  Target notice ID '{target_notice_id}' does not exist in searchable data")
            print(f"        (It may be too old, archived, or the ID may be incorrect)")
        
        if real_id_tests:
            print(f"   ✅ Notice ID search functionality works (tested with known IDs)")
        
        # Parameter analysis
        if camel_case_success and not lowercase_success:
            print(f"   ✅ Use 'noticeId' (camelCase) parameter - 'noticeid' (lowercase) doesn't work")
        elif lowercase_success and not camel_case_success:
            print(f"   ✅ Use 'noticeid' (lowercase) parameter - 'noticeId' (camelCase) doesn't work")
        elif camel_case_success and lowercase_success:
            print(f"   ✅ Both 'noticeId' and 'noticeid' parameters work")
        
        # URL analysis
        if prod_url_success and no_prod_success:
            print(f"   ✅ Both /prod/ and non-/prod/ URLs work")
        elif prod_url_success:
            print(f"   ✅ Use /prod/ in URL - non-/prod/ URLs don't work")
        elif no_prod_success:
            print(f"   ✅ Don't use /prod/ in URL - /prod/ URLs don't work")
    
    else:
        print(f"   ❌ SAM.gov API is not accessible or API key is invalid")
        print(f"   ❌ Cannot determine working parameters")

if __name__ == "__main__":
    test_sam_api_with_curl()