#!/usr/bin/env python3

"""
Test script to demonstrate the enhanced AI agent with Tavily search and LangSmith integration.
"""

from agent import SimpleAIAgent
import os
from dotenv import load_dotenv

load_dotenv()

def test_agent():
    """Test the enhanced agent with different types of queries"""
    
    # Check if API keys are available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Missing TAVILY_API_KEY")
        return
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ Missing LANGSMITH_API_KEY")
        return
    
    print("🤖 Enhanced AI Agent Test")
    print("=" * 50)
    
    # Initialize the agent
    try:
        agent = SimpleAIAgent()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "Memory Test (No Search)",
            "query": "Hello, my name is John and I love programming",
            "should_search": False
        },
        {
            "name": "Current Information Test (Should Search)",
            "query": "What's the latest news about AI today?",
            "should_search": True
        },
        {
            "name": "Memory Recall Test (No Search)",
            "query": "Do you remember my name?",
            "should_search": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print("-" * 30)
        print(f"Query: {test_case['query']}")
        
        try:
            response = agent.chat(test_case["query"], user_name="TestUser", thread_id="test_session")
            print(f"Response: {response}")
            
            # Check if the agent is working properly
            if response and len(response) > 10:
                print("✅ Response received")
            else:
                print("❌ No meaningful response")
                
        except Exception as e:
            print(f"❌ Error during chat: {e}")
    
    print("\n🎯 Graph Structure:")
    print("=" * 30)
    print("process_input → decide_search → [web_search] → generate_response → update_memory → END")
    print("\nNodes:")
    print("1. process_input - Initialize state")
    print("2. decide_search - Analyze if web search needed")
    print("3. web_search - Perform Tavily search (conditional)")
    print("4. generate_response - Generate response with LLM")
    print("5. update_memory - Store interaction in memory")
    
    print("\n📊 State Information:")
    print("- messages: List of HumanMessage/AIMessage objects")
    print("- memory: Dictionary of previous interactions")
    print("- user_name: User identifier")
    print("- search_results: List of web search results")
    print("- needs_search: Boolean flag for search decision")
    
    print("\n🔍 LangSmith Integration:")
    print("- All nodes are decorated with @traceable")
    print("- Full workflow visibility in LangSmith dashboard")
    print("- Environment: LANGSMITH_TRACING_V2=true")

if __name__ == "__main__":
    test_agent()