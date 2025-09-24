#!/usr/bin/env python3
"""Debug script to understand Browser Use AgentHistoryList structure"""

import asyncio
import os
import logging
from browser_use import Agent, Browser, ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_browser_use():
    """Test what Browser Use returns after timeout"""
    
    logger.info("Testing Browser Use return values...")
    
    # Initialize LLM  
    llm = ChatOpenAI(
        model="gpt-5",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create browser
    browser = Browser(headless=True)
    
    # Create agent
    agent = Agent(
        task="Go to https://example.com and extract the page title",
        llm=llm,
        browser=browser,
        use_vision=False,
        max_steps=3
    )
    
    try:
        # Run with short timeout
        try:
            result = await asyncio.wait_for(agent.run(), timeout=30)
            logger.info(f"Normal completion - Result type: {type(result)}")
            logger.info(f"Result: {result}")
        except asyncio.TimeoutError:
            logger.info("Timeout occurred, checking agent.history...")
            result = agent.history
        
        # Analyze the result structure
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result attributes: {dir(result)}")
        
        if hasattr(result, 'extracted_content'):
            extracted = result.extracted_content()
            logger.info(f"extracted_content() type: {type(extracted)}")
            logger.info(f"extracted_content() value: {extracted}")
            
        if hasattr(result, 'final_result'):
            final = result.final_result()
            logger.info(f"final_result() type: {type(final)}")
            logger.info(f"final_result() value: {final}")
            
        if hasattr(result, 'action_results'):
            actions = result.action_results()
            logger.info(f"action_results() type: {type(actions)}")
            logger.info(f"Number of action results: {len(actions) if actions else 0}")
            for i, action in enumerate(actions[:3] if actions else []):
                logger.info(f"Action {i}: {type(action)} - {action}")
        
        # Check for any extractable text content
        logger.info(f"String representation length: {len(str(result))}")
        logger.info(f"First 500 chars of string: {str(result)[:500]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None
    finally:
        await browser.close()

if __name__ == "__main__":
    result = asyncio.run(debug_browser_use())
    print(f"\nFinal result type: {type(result)}")