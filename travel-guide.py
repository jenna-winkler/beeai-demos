import os
from collections.abc import AsyncGenerator

from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.models import CitationMetadata, TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, Server

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

server = Server()

conversation_memories = {}


class TrajectoryCapture:
    """Captures trajectory steps for display"""
    def __init__(self):
        self.steps = []
    
    def write(self, message: str) -> int:
        self.steps.append(message.strip())
        return len(message)


class TrackedTool:
    """Base class for tool tracking"""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.results = []
    
    def add_result(self, result):
        self.results.append(result)


class TrackedDuckDuckGoTool(DuckDuckGoSearchTool):
    """DuckDuckGo tool with result tracking"""
    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker
    
    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(('DuckDuckGo', result))
        return result


class TrackedWikipediaTool(WikipediaTool):
    """Wikipedia tool with result tracking"""
    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker
    
    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(('Wikipedia', result))
        return result


class TrackedOpenMeteoTool(OpenMeteoTool):
    """Weather tool with result tracking"""
    def __init__(self, tracker: TrackedTool):
        super().__init__()
        self.tracker = tracker
    
    async def _run(self, input_data, options, context):
        result = await super()._run(input_data, options, context)
        self.tracker.add_result(('OpenMeteo', result))
        return result


def get_session_id(context: Context) -> str:
    """Extract session ID from context, fallback to default if not available"""
    session_id = getattr(context, 'session_id', None)
    if not session_id:
        session_id = getattr(context, 'headers', {}).get('session-id', 'default')
    return str(session_id)


def get_or_create_memory(session_id: str) -> UnconstrainedMemory:
    """Get existing memory for session or create new one"""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = UnconstrainedMemory()
    return conversation_memories[session_id]


@server.agent(
    name="travel_guide",
    description="Comprehensive travel guide agent that provides personalized recommendations with weather, local information, and current search results. Features dynamic citations and trajectory tracking.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi! I'm your Travel Guide - here to help plan trips, check weather, and recommend restaurants. Where to?",
                display_name="Travel Guide",
                tools=[
                    AgentToolInfo(
                        name="Think", 
                        description="Advanced reasoning and analysis for travel planning, itinerary optimization, and recommendation personalization based on your preferences and constraints."
                    ),
                    AgentToolInfo(
                        name="Wikipedia", 
                        description="Search comprehensive information about destinations, attractions, history, culture, and local knowledge from Wikipedia's vast database."
                    ),
                    AgentToolInfo(
                        name="Weather", 
                        description="Get current weather conditions, forecasts, and climate information for any destination to help with travel planning and packing decisions."
                    ),
                    AgentToolInfo(
                        name="DuckDuckGo", 
                        description="Search for current information about restaurants, hotels, events, transportation, and real-time travel updates from across the web."
                    ),
                ]
            )
        ),
        author={
            "name": "Jenna Winkler"
        },
        recommended_models=[
            "granite3.3:8b-beeai"
        ],
        tags=["Travel", "Planning", "Research"],
        framework="BeeAI",
        programming_language="Python",
        license="Apache 2.0"
    )
)
async def travel_guide(input: list[Message], context: Context) -> AsyncGenerator:
    """
    Comprehensive travel guide agent that combines:
    - Dynamic citations from search results
    - Trajectory tracking for transparency
    - Multi-tool integration for comprehensive travel planning
    """
    
    user_message = input[-1].parts[0].content if input else "Hello"
    session_id = get_session_id(context)
    
    tool_tracker = TrackedTool("travel_guide")
    trajectory = TrajectoryCapture()
    
    session_memory = get_or_create_memory(session_id)
    
    yield MessagePart(metadata=TrajectoryMetadata(
        message=f"🌍 Travel Guide processing: '{user_message}'"
    ))
    
    try:
        await session_memory.add(UserMessage(user_message))
        
        tracked_duckduckgo = TrackedDuckDuckGoTool(tool_tracker)
        tracked_wikipedia = TrackedWikipediaTool(tool_tracker)
        tracked_weather = TrackedOpenMeteoTool(tool_tracker)
        
        agent = RequirementAgent(
            llm=ChatModel.from_name("ollama:granite3.3:8b"),
            memory=session_memory,  
            tools=[
                ThinkTool(), 
                tracked_wikipedia, 
                tracked_weather, 
                tracked_duckduckgo
            ],
            requirements=[
                ConditionalRequirement(
                    ThinkTool, 
                    force_at_step=1, 
                    force_after=Tool, 
                    consecutive_allowed=False,
                    max_invocations=3 
                ),
                ConditionalRequirement(
                    tracked_wikipedia,
                    max_invocations=1,
                    consecutive_allowed=False
                ),
                ConditionalRequirement(
                    tracked_weather,
                    max_invocations=1,
                    consecutive_allowed=False
                ),
                ConditionalRequirement(
                    tracked_duckduckgo,
                    max_invocations=1, 
                    consecutive_allowed=False
                )
            ],
            instructions="""You are a comprehensive travel guide assistant. Your goal is to provide helpful, accurate, and personalized travel recommendations.

            IMPORTANT WORKFLOW:
            1. Think about the user's request first
            2. Gather information efficiently using tools (don't repeat searches unnecessarily)
            3. Provide a comprehensive final answer based on the information gathered

            For travel planning queries:
            1. First, think about what information would be most helpful
            2. Use Wikipedia for destination background, history, and general information (search once per destination)
            3. Use OpenMeteo for current weather conditions and forecasts (search once per location)
            4. Use DuckDuckGo for current restaurant recommendations, events, hotels, and real-time information (be specific in searches)
            
            Always provide in your final answer:
            - Practical travel advice
            - Local insights and cultural tips
            - Weather-appropriate recommendations
            - Current and up-to-date information
            - Personalized suggestions based on user preferences
            
            Be conversational, helpful, and enthusiastic about travel while providing accurate information. 
            
            CRITICAL: Once you have gathered sufficient information, provide your comprehensive final answer. Do not continue searching unnecessarily."""
        )
        
        yield MessagePart(metadata=TrajectoryMetadata(
            message=f"🛠️ Travel Guide initialized with Think, Wikipedia, Weather, and Search tools"
        ))
        
        response = await agent.run(
            user_message,
            execution=AgentExecutionConfig(
                max_iterations=10,  
                max_retries_per_step=2, 
                total_max_retries=5 
            )
        ).middleware(
            GlobalTrajectoryMiddleware(target=trajectory, included=[Tool])
        )
        
        response_text = response.answer.text
        
        await session_memory.add(AssistantMessage(response_text))
        
        for i, step in enumerate(trajectory.steps):
            if step.strip():
                tool_name = None
                if "ThinkTool" in step:
                    tool_name = "Think"
                elif "WikipediaTool" in step:
                    tool_name = "Wikipedia"  
                elif "OpenMeteoTool" in step:
                    tool_name = "Weather"
                elif "DuckDuckGo" in step:
                    tool_name = "DuckDuckGo"
                    
                yield MessagePart(metadata=TrajectoryMetadata(
                    message=f"Step {i+1}: {step}",
                    tool_name=tool_name
                ))
        
        yield MessagePart(content=response_text)
        
        citation_count = 0
        total_citations = 0

        for tool_name, tool_output in tool_tracker.results:
            if total_citations >= 10:  # Global limit
                break
            
            tool_citation_count = 0  # Track citations per tool
            max_citations_per_tool = 2  # Limit each tool to 2 citations
            
            if tool_name == 'Wikipedia' and hasattr(tool_output, 'results') and tool_output.results:
                for result in tool_output.results:
                    if tool_citation_count >= max_citations_per_tool or total_citations >= 10:
                        break
                    title_words = result.title.split()
                    for word in title_words:
                        if word.lower() in response_text.lower() and len(word) > 3:
                            start_idx = response_text.lower().find(word.lower())
                            if start_idx != -1:
                                yield MessagePart(
                                    metadata=CitationMetadata(
                                        url=result.url,
                                        title=result.title,
                                        description=result.description[:100] + "..." if len(result.description) > 100 else result.description,
                                        start_index=start_idx,
                                        end_index=start_idx + len(word)
                                    )
                                )
                                tool_citation_count += 1
                                total_citations += 1
                                break
                                
            elif tool_name == 'DuckDuckGo' and hasattr(tool_output, 'results') and tool_output.results:
                for result in tool_output.results:
                    if tool_citation_count >= max_citations_per_tool or total_citations >= 10:
                        break
                    title_words = result.title.split()
                    for word in title_words:
                        if word.lower() in response_text.lower() and len(word) > 4:
                            start_idx = response_text.lower().find(word.lower())
                            if start_idx != -1:
                                yield MessagePart(
                                    metadata=CitationMetadata(
                                        url=result.url,
                                        title=result.title,
                                        description=result.description[:100] + "..." if len(result.description) > 100 else result.description,
                                        start_index=start_idx,
                                        end_index=start_idx + len(word)
                                    )
                                )
                                tool_citation_count += 1
                                total_citations += 1
                                break
                                
            elif tool_name == 'OpenMeteo':
                if tool_citation_count >= max_citations_per_tool or total_citations >= 10:
                    continue
                weather_words = ["weather", "temperature", "warm", "cool", "forecast", "conditions", "climate", "rain", "sunny", "cloudy"]
                for word in weather_words:
                    if tool_citation_count >= max_citations_per_tool or total_citations >= 10:
                        break
                    if word in response_text.lower():
                        start_idx = response_text.lower().find(word)
                        yield MessagePart(
                            metadata=CitationMetadata(
                                url="https://open-meteo.com/",
                                title="Open-Meteo Weather API",
                                description="Real-time weather data and forecasts",
                                start_index=start_idx,
                                end_index=start_idx + len(word)
                            )
                        )
                        tool_citation_count += 1
                        total_citations += 1
                        break
        
        yield MessagePart(metadata=TrajectoryMetadata(
            message="✅ Travel Guide completed successfully with citations"
        ))
        
    except Exception as e:
        yield MessagePart(metadata=TrajectoryMetadata(
            message=f"❌ Error: {str(e)}"
        ))
        yield MessagePart(content=f"🚨 Sorry, I encountered an error while planning your trip: {str(e)}")


def run():
    """Entry point for the server."""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
