#!/usr/bin/env python3

from agent import SimpleAIAgent
import os

def main():
    print("🤖 Simple AI Agent with Memory")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("❌ Please set your OPENAI_API_KEY in the .env file")
        print("Edit the .env file and replace 'your_openai_api_key_here' with your actual OpenAI API key")
        return
    
    agent = SimpleAIAgent()
    user_name = input("What's your name? ") or "User"
    thread_id = "demo_session"
    
    print(f"\nHello {user_name}! I'm your AI assistant with memory.")
    print("I'll remember our conversation throughout this session.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        try:
            user_input = input(f"{user_name}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting!")
                break
            
            if not user_input:
                continue
            
            print("🤔 Thinking...")
            response = agent.chat(user_input, user_name, thread_id)
            print(f"🤖 Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'quit' to exit.\n")

if __name__ == "__main__":
    main()