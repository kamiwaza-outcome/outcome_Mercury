#!/usr/bin/env python3
"""Debug script to understand what browser-use actually returns"""

import asyncio
import os
import logging
import json
from pprint import pprint
from browser_use import Agent, Browser, ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_browser_use():
    """Debug what browser-use actually returns"""
    
    logger.info("Debugging browser-use result structure...")
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create browser
    browser = Browser(headless=True)
    
    # Create agent with simple task
    agent = Agent(
        task="Go to https://example.com and get the page title",
        llm=llm,
        browser=browser,
        use_vision=False
    )
    
    try:
        # Run the agent
        result = await agent.run()
        
        logger.info("=== BROWSER-USE RESULT DEBUG ===")
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result class: {result.__class__}")
        logger.info(f"Result dir: {dir(result)}")
        
        # Check if it has the methods we expect
        if hasattr(result, 'action_results'):
            logger.info(f"Has action_results method: {hasattr(result, 'action_results')}")
            action_results = result.action_results()
            logger.info(f"Action results type: {type(action_results)}")
            logger.info(f"Action results length: {len(action_results)}")
            
            for i, action_result in enumerate(action_results):
                logger.info(f"Action {i} type: {type(action_result)}")
                logger.info(f"Action {i} dir: {dir(action_result)}")
                if hasattr(action_result, 'extracted_content'):
                    logger.info(f"Action {i} extracted_content: {action_result.extracted_content}")
                if hasattr(action_result, 'text'):
                    logger.info(f"Action {i} text: {action_result.text}")
                if hasattr(action_result, 'result'):
                    logger.info(f"Action {i} result: {action_result.result}")
        
        if hasattr(result, 'model_actions'):
            logger.info(f"Has model_actions method: {hasattr(result, 'model_actions')}")
            model_actions = result.model_actions()
            logger.info(f"Model actions type: {type(model_actions)}")
            logger.info(f"Model actions length: {len(model_actions)}")
        
        if hasattr(result, 'history'):
            logger.info(f"Has history attribute: {hasattr(result, 'history')}")
            logger.info(f"History type: {type(result.history)}")
            logger.info(f"History length: {len(result.history)}")
            
            # Look at last history entry
            if result.history:
                last_history = result.history[-1]
                logger.info(f"Last history type: {type(last_history)}")
                logger.info(f"Last history dir: {dir(last_history)}")
                if hasattr(last_history, 'result'):
                    logger.info(f"Last history result: {last_history.result}")
        
        # Try to convert to string
        result_str = str(result)
        logger.info(f"String representation length: {len(result_str)}")
        logger.info(f"String representation preview: {result_str[:500]}")
        
        return result
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        await browser.close()

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in environment")
        exit(1)
    
    result = asyncio.run(debug_browser_use())
    if result:
        print("\n✅ Debug completed - check logs above")
    else:
        print("\n❌ Debug failed - check logs above")