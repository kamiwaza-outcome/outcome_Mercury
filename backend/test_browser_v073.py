#!/usr/bin/env python3
"""Test browser-use v0.7.3 with screenshot patch"""

import asyncio
import os
import logging
from browser_use import Agent, Browser
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_browser_use():
    """Test browser-use with the v0.7.3 patch"""
    
    # Import the patch function from browser_scraper
    from services.browser_scraper import patch_browser_use_v073
    
    # Apply the patch
    patch_browser_use_v073()
    
    logger.info("Testing browser-use v0.7.3 with screenshot patch...")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=1.0,
        max_completion_tokens=4096
    )
    
    # Create browser
    browser = Browser(headless=True)
    
    # Create agent with use_vision=False
    agent = Agent(
        task="Go to https://sam.gov and extract the page title",
        llm=llm,
        browser=browser,
        use_vision=False  # This should now skip screenshots
    )
    
    try:
        # Run the agent
        result = await agent.run(max_steps=3)
        logger.info(f"Success! Result: {result}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    finally:
        await browser.close()

if __name__ == "__main__":
    success = asyncio.run(test_browser_use())
    if success:
        print("\n✅ Test passed - no screenshot timeout!")
    else:
        print("\n❌ Test failed - check logs above")