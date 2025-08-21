# Enhanced AI Agent with Memory & Web Search

A sophisticated conversational AI agent built with LangChain and LangGraph featuring persistent memory, intelligent web search capabilities, and automatic conversation compaction. The agent uses a 7-node state machine with smart search detection and memory management.

## Features

- **Conversational Memory**: Persistent memory across sessions with automatic compaction
- **Intelligent Web Search**: Automatic detection of queries requiring current information using Tavily
- **LangGraph State Machine**: 7-node workflow with conditional routing and memory management
- **Memory Compaction**: Automatic conversation summarization every 5 turns to optimize memory
- **LangSmith Integration**: Full tracing and observability for debugging and monitoring
- **LangGraph Studio Support**: Visual workflow debugging and development
- **Command-line Interface**: Interactive CLI with user-friendly experience

## Quick Start

### Environment Setup
1. Activate the virtual environment:
   ```bash
   source langchain_agent_env/bin/activate
   ```

2. Set required API keys in `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional but recommended
   ```

### Running the Agent

#### CLI Mode (Direct)
```bash
python main.py
```

#### LangGraph Studio Mode
```bash
./start_studio.sh
# OR
langgraph dev
```
Studio UI available at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

#### Testing
```bash
python test_enhanced_agent.py
```

## Architecture

### 7-Node State Machine Workflow
1. **process_input**: Initialize state variables and validate input
2. **decide_search**: Analyze user query for search indicators using keywords and patterns
3. **web_search**: Conditional Tavily web search for current information
4. **generate_response**: Generate contextual response using OpenAI with memory and search context
5. **check_compact**: Evaluate if memory compaction is needed (every 5 turns)
6. **compact_memory**: Summarize conversation history and prune old interactions
7. **update_memory**: Store new interaction in persistent memory

### Enhanced State Management
The `AgentState` includes:
- `messages`: List of conversation messages (HumanMessage/AIMessage)
- `memory`: Dictionary storing interactions and conversation summaries
- `user_name`: User identifier for personalization
- `search_results`: Web search results from Tavily
- `needs_search`: Boolean flag for search requirement detection
- `turn_count`: Counter for triggering memory compaction
- `needs_compact`: Boolean flag for compaction decisions
- `conversation_summary`: Condensed conversation history

### Intelligent Search Detection
The agent automatically detects when queries require current information by analyzing:
- **Keywords**: "current", "latest", "recent", "now", "today", "news", "price", etc.
- **Patterns**: Questions about costs, prices, releases, recent events
- **Context**: Real-time data requests and breaking news queries

### Memory Management
- **Persistent Storage**: Uses LangGraph MemorySaver for cross-session memory
- **Automatic Compaction**: Summarizes conversations every 5 turns
- **Smart Pruning**: Maintains conversation summaries while removing old detailed interactions
- **Context Preservation**: Retains important user preferences and context

## Dependencies

Key packages (managed in virtual environment):
- `langchain ^0.3.27`: Core LangChain functionality
- `langgraph ^0.6.6`: State machine and workflow management
- `langchain-openai ^0.3.30`: OpenAI GPT integration
- `langchain-tavily ^0.2.11`: Web search capabilities
- `python-dotenv ^1.1.1`: Environment variable management

## Configuration

The agent supports configuration through:
- Environment variables in `.env` file
- `langgraph.json` for Studio integration
- Model selection (default: gpt-3.5-turbo, configurable via AI_MODEL env var)

## LangSmith Integration

All workflow nodes are instrumented with `@traceable` decorators providing:
- Complete execution tracing
- Performance monitoring
- Debug visibility
- Workflow analysis in LangSmith dashboard