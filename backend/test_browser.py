#!/usr/bin/env python3
import asyncio
import os
from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_browser_scrape():
    """Test Browser Use with the NARWHAL RFP URL"""
    url = "https://sam.gov/opp/710637f47916457280ca93dbb73c2b41/view"
    
    print(f"Testing Browser Use with URL: {url}")
    print("This may take a few minutes...")
    
    llm = ChatOpenAI(
        model="gpt-5",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=1.0,  # GPT-5 only supports temperature=1
        max_completion_tokens=16384  # Use max_completion_tokens for GPT-5
    )
    
    # Create headless browser controller
    controller = Controller(headless=True, keep_open=False)
    
    agent = Agent(
        task=(
            f"Go to this SAM.gov page: {url}\n"
            "Extract and return:\n"
            "1. The title of the opportunity\n"
            "2. The description\n"
            "3. The agency name\n"
            "4. What type of opportunity this is (AI research, construction, etc.)\n"
            "5. Any key requirements\n"
            "If the page doesn't load or shows an error, report that."
        ),
        llm=llm,
        controller=controller
    )
    
    try:
        result = await agent.run()
        print("\n=== Browser Use Result ===")
        print(result)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_browser_scrape())