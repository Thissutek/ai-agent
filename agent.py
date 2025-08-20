from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langsmith import traceable
import os
import re
from dotenv import load_dotenv

load_dotenv()

class AgentState(MessagesState):
    memory: Dict[str, Any]
    user_name: str
    search_results: List[Dict[str, Any]]
    needs_search: bool
    turn_count: int
    needs_compact: bool
    conversation_summary: str

class SimpleAIAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.search_tool = TavilySearch(
            max_results=3
        )
        
        self.memory = MemorySaver()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with memory capabilities and web search access.
            You remember previous conversations and can search for current information when needed.
            
            User's name: {user_name}
            
            Memory context: {memory_context}
            
            Search results (if available): {search_context}
            
            Be conversational, helpful, and remember details from previous interactions.
            When you have search results, integrate them naturally into your response."""),
            ("placeholder", "{messages}")
        ])
        
        self.graph = self._create_graph()
    
    @classmethod
    def create_graph(cls):
        """Factory method to create the graph for LangGraph Studio"""
        agent = cls()
        return agent.graph
    
    def _create_graph(self, use_checkpointer: bool = True) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("decide_search", self._decide_search)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("check_compact", self._check_compact)
        workflow.add_node("compact_memory", self._compact_memory)
        workflow.add_node("update_memory", self._update_memory)
        
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "decide_search")
        
        # Conditional routing from decide_search
        workflow.add_conditional_edges(
            "decide_search",
            self._route_search,
            {
                "search": "web_search",
                "no_search": "generate_response"
            }
        )
        
        workflow.add_edge("web_search", "generate_response")
        workflow.add_edge("generate_response", "check_compact")
        
        # Conditional routing from check_compact
        workflow.add_conditional_edges(
            "check_compact",
            self._route_compact,
            {
                "compact": "compact_memory",
                "no_compact": "update_memory"
            }
        )
        
        workflow.add_edge("compact_memory", "update_memory")
        workflow.add_edge("update_memory", END)
        
        if use_checkpointer:
            return workflow.compile(checkpointer=self.memory)
        else:
            return workflow.compile()
    
    @traceable(name="process_input")
    def _process_input(self, state: AgentState) -> AgentState:
        # Initialize search-related state if not present
        if "search_results" not in state:
            state["search_results"] = []
        if "needs_search" not in state:
            state["needs_search"] = False
        # Initialize compact-related state
        if "turn_count" not in state:
            state["turn_count"] = 0
        if "needs_compact" not in state:
            state["needs_compact"] = False
        if "conversation_summary" not in state:
            state["conversation_summary"] = ""
        return state
    
    @traceable(name="decide_search")
    def _decide_search(self, state: AgentState) -> AgentState:
        """Analyze the user's message to determine if web search is needed"""
        if not state["messages"]:
            state["needs_search"] = False
            return state
            
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            user_input = last_message.content.lower()
            
            # Keywords that suggest current information is needed
            search_indicators = [
                "current", "latest", "recent", "now", "today", "2024", "2025",
                "news", "price", "stock", "weather", "what's happening",
                "update", "status", "live", "real-time", "breaking"
            ]
            
            # Questions that often require current information
            question_patterns = [
                r"what.*(happening|going on)",
                r"how much.*cost",
                r"what.*price",
                r"when.*release",
                r"who.*recently"
            ]
            
            needs_search = any(indicator in user_input for indicator in search_indicators)
            needs_search = needs_search or any(re.search(pattern, user_input) for pattern in question_patterns)
            
            state["needs_search"] = needs_search
        
        return state
    
    def _route_search(self, state: AgentState) -> str:
        """Route to search or skip search based on decision"""
        return "search" if state.get("needs_search", False) else "no_search"
    
    @traceable(name="web_search")
    def _web_search(self, state: AgentState) -> AgentState:
        """Perform web search using Tavily"""
        if not state["messages"]:
            return state
            
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            try:
                # Use run method to get search results
                search_result = self.search_tool.run(last_message.content)
                # Convert to expected format
                state["search_results"] = [{"content": search_result, "title": "Search Results", "url": "tavily://search"}]
            except Exception as e:
                print(f"Search error: {e}")
                state["search_results"] = []
        
        return state
    
    @traceable(name="generate_response")
    def _generate_response(self, state: AgentState) -> AgentState:
        memory_context = self._format_memory(state.get("memory", {}))
        search_context = self._format_search_results(state.get("search_results", []))
        
        formatted_prompt = self.prompt.format_messages(
            user_name=state.get("user_name", "User"),
            memory_context=memory_context,
            search_context=search_context,
            messages=state["messages"]
        )
        
        response = self.llm.invoke(formatted_prompt)
        
        # Increment turn count after AI response
        new_turn_count = state.get("turn_count", 0) + 1
        print(f"DEBUG: AI response generated, turn_count now: {new_turn_count}")
        
        # Return the new message - MessagesState will handle appending automatically
        return {
            "messages": [response],
            "turn_count": new_turn_count
        }
    
    @traceable(name="update_memory")
    def _update_memory(self, state: AgentState) -> AgentState:
        if len(state["messages"]) >= 2:
            last_human_msg = None
            last_ai_msg = None
            
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and last_ai_msg is None:
                    last_ai_msg = msg
                elif isinstance(msg, HumanMessage) and last_human_msg is None:
                    last_human_msg = msg
                
                if last_human_msg and last_ai_msg:
                    break
            
            if last_human_msg and last_ai_msg:
                memory_key = f"interaction_{len(state.get('memory', {}))}"
                if "memory" not in state:
                    state["memory"] = {}
                
                state["memory"][memory_key] = {
                    "user_input": last_human_msg.content,
                    "ai_response": last_ai_msg.content,
                    "had_search": len(state.get("search_results", [])) > 0,
                    "timestamp": "now"
                }
        
        return state
    
    @traceable(name="check_compact")
    def _check_compact(self, state: AgentState) -> AgentState:
        """Check if conversation needs to be compacted after 5 turns"""
        turn_count = state.get("turn_count", 0)
        needs_compact = turn_count > 0 and turn_count % 5 == 0
        print(f"DEBUG: turn_count={turn_count}, needs_compact={needs_compact}")
        state["needs_compact"] = needs_compact
        return state
    
    def _route_compact(self, state: AgentState) -> str:
        """Route to compact or skip compact based on turn count"""
        return "compact" if state.get("needs_compact", False) else "no_compact"
    
    @traceable(name="compact_memory")
    def _compact_memory(self, state: AgentState) -> AgentState:
        """Summarize the conversation to save memory"""
        print(f"DEBUG: compact_memory called with {len(state['messages'])} messages")
        
        # In Studio, we can't rely on full message history being available
        # Instead, we'll use the memory to track interactions and summarize from there
        memory = state.get("memory", {})
        if len(memory) < 3:  # Need at least some interactions to summarize
            print("DEBUG: Not enough memory interactions to summarize")
            return state
        
        # Create summarization prompt
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are tasked with creating a concise summary of a conversation between a user and an AI assistant.
            Focus on:
            1. Key topics discussed
            2. Important information shared
            3. User preferences or context that should be remembered
            4. Any search results that were important
            
            Keep the summary under 300 words and maintain the most important context for future interactions."""),
            ("user", "Please summarize this conversation:\n\n{conversation}")
        ])
        
        # Format conversation from memory for summarization
        conversation_text = ""
        for key, interaction in memory.items():
            if interaction.get('type') != 'conversation_summary':  # Don't include old summaries
                conversation_text += f"User: {interaction.get('user_input', '')}\n"
                conversation_text += f"AI: {interaction.get('ai_response', '')}\n"
                if interaction.get('had_search', False):
                    conversation_text += f"(Used web search)\n"
                conversation_text += "\n"
        
        try:
            summary_messages = summary_prompt.format_messages(conversation=conversation_text)
            summary_response = self.llm.invoke(summary_messages)
            state["conversation_summary"] = summary_response.content
            
            # Clear old interactions and keep only summary + recent interactions
            old_interactions = [k for k in memory.keys() if not k.startswith('summary_')]
            
            # Keep only the last 2 interactions and add summary to memory  
            if len(old_interactions) > 2:
                # Sort by timestamp or key order and keep last 2
                recent_keys = sorted(old_interactions)[-2:]
                for key in old_interactions:
                    if key not in recent_keys:
                        del state["memory"][key]
                
                # Add summary to memory
                if "memory" not in state:
                    state["memory"] = {}
                
                summary_key = f"summary_{len([k for k in state['memory'].keys() if k.startswith('summary_')])}"
                state["memory"][summary_key] = {
                    "type": "conversation_summary",
                    "content": summary_response.content,
                    "turn_count": state.get("turn_count", 0),
                    "timestamp": "now"
                }
                print(f"DEBUG: Created conversation summary: {summary_response.content[:100]}...")
                
        except Exception as e:
            print(f"Compact error: {e}")
        
        return state
    
    def _format_memory(self, memory: Dict[str, Any]) -> str:
        if not memory:
            return "No previous interactions."
        
        memory_str = "Previous interactions:\n"
        
        # Show conversation summaries first
        for key, interaction in memory.items():
            if interaction.get('type') == 'conversation_summary':
                memory_str += f"[SUMMARY up to turn {interaction.get('turn_count', 0)}]: {interaction['content']}\n\n"
        
        # Then show recent individual interactions
        for key, interaction in memory.items():
            if interaction.get('type') != 'conversation_summary':
                memory_str += f"- User said: {interaction['user_input'][:100]}...\n"
                memory_str += f"  AI responded: {interaction['ai_response'][:100]}...\n"
                if interaction.get('had_search', False):
                    memory_str += f"  (Used web search)\n"
                memory_str += "\n"
        
        return memory_str
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        if not search_results:
            return "No web search performed."
        
        search_str = "Web search results:\n"
        for i, result in enumerate(search_results[:3], 1):  # Limit to top 3 results
            if isinstance(result, dict):
                title = result.get('title', 'No title')
                content = result.get('content', result.get('snippet', 'No content'))
                # Ensure content is a string and then truncate
                content_str = str(content)[:200] if content else 'No content'
                url = result.get('url', 'No URL')
                search_str += f"{i}. {title}\n   {content_str}...\n   Source: {url}\n\n"
        
        return search_str
    
    @traceable(name="chat_session")
    def chat(self, message: str, user_name: str = "User", thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get existing state or create initial state
        try:
            current_state = self.graph.get_state(config)
            if current_state.values:
                # Use existing state and add new message
                state = current_state.values.copy()
                state["messages"].append(HumanMessage(content=message))
                state["user_name"] = user_name  # Update user name if changed
            else:
                raise ValueError("No existing state")
        except:
            # Create initial state for new thread
            state = {
                "messages": [HumanMessage(content=message)],
                "memory": {},
                "user_name": user_name,
                "search_results": [],
                "needs_search": False,
                "turn_count": 0,
                "needs_compact": False,
                "conversation_summary": ""
            }
        
        result = self.graph.invoke(state, config)
        
        return result["messages"][-1].content if result["messages"] else "No response generated."


# Module-level graph export for LangGraph Studio
def create_agent_graph():
    """Create and return the agent graph for LangGraph Studio
    
    Note: Studio manages persistence automatically, so memory/compacting
    features work differently than in direct usage with checkpointer.
    """
    agent = SimpleAIAgent()
    # Studio doesn't use checkpointer - it handles persistence differently
    agent.graph = agent._create_graph(use_checkpointer=False)
    return agent.graph

# Export the graph at module level for Studio
agent_graph = create_agent_graph()