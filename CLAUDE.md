# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Enhanced AI agent built with LangChain and LangGraph featuring conversational memory, web search capabilities, and memory compaction. The agent uses a 7-node state machine with intelligent search detection and automatic conversation summarization.

## Development Setup

### Environment Setup
1. Activate the virtual environment:
   ```bash
   source langchain_agent_env/bin/activate
   ```

2. Ensure all API keys are set in `.env` file:
   - `OPENAI_API_KEY`: Required for GPT responses
   - `TAVILY_API_KEY`: Required for web search functionality
   - `LANGSMITH_API_KEY`: Required for tracing (optional but recommended)

### Running the Agent

#### CLI Mode (Direct)
```bash
python main.py
```

#### LangGraph Studio Mode
```bash
./start_studio.sh
# OR
source langchain_agent_env/bin/activate && langgraph dev
```
Studio UI available at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

#### Testing
```bash
python test_enhanced_agent.py
```

### Dependencies
Key dependencies managed in virtual environment:
- `langchain ^0.3.27`: Core LangChain functionality
- `langgraph ^0.6.6`: State machine and workflow management  
- `langchain-openai ^0.3.30`: OpenAI integration
- `langchain-tavily ^0.2.11`: Web search capabilities
- `python-dotenv ^1.1.1`: Environment variable management

## Architecture

### Core Files
- **main.py**: CLI entry point with user interaction loop
- **agent.py**: `SimpleAIAgent` class with LangGraph workflow and Studio integration
- **test_enhanced_agent.py**: Comprehensive testing script
- **langgraph.json**: Studio configuration defining graph export
- **start_studio.sh**: Convenience script for Studio startup

### Enhanced LangGraph State Machine
The agent implements a 7-node workflow:
1. **process_input**: Initialize state variables and validate input
2. **decide_search**: Analyze user query for search indicators
3. **web_search**: Conditional Tavily web search based on query analysis
4. **generate_response**: Generate response using OpenAI with context
5. **check_compact**: Evaluate if memory compaction is needed (every 5 turns)
6. **compact_memory**: Summarize conversation and prune old interactions
7. **update_memory**: Store new interaction in persistent memory

### Enhanced State Management
`AgentState` TypedDict contains:
- `messages`: List of `HumanMessage`/`AIMessage` objects
- `memory`: Dictionary storing interactions and summaries
- `user_name`: User identifier for personalization
- `search_results`: List of web search results from Tavily
- `needs_search`: Boolean flag indicating search requirement
- `turn_count`: Counter for triggering memory compaction
- `needs_compact`: Boolean flag for compaction decision
- `conversation_summary`: Condensed conversation history

### Memory & Search Systems
- **Memory**: LangGraph `MemorySaver` with automatic compaction every 5 turns
- **Search Decision Logic**: Keyword and pattern matching for current information needs
- **Search Integration**: Tavily `get_search_context()` for AI-optimized results
- **LangSmith Tracing**: Full workflow observability with `@traceable` decorators

## Key Implementation Details

- Uses OpenAI GPT-3.5-turbo (configurable in .env as `AI_MODEL=gpt-4o-mini`)
- Intelligent search triggers: current events, prices, recent news, real-time data
- Memory compaction preserves conversation summaries while pruning old interactions
- Studio compatibility through module-level graph export (`agent_graph`)
- Comprehensive error handling for API failures and missing credentials