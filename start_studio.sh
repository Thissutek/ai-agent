#!/bin/bash

# Start LangGraph Studio for AI Agent development
echo "Starting LangGraph Studio..."
echo "Studio UI will be available at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
echo "API will be available at: http://127.0.0.1:2024"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

source langchain_agent_env/bin/activate && langgraph dev