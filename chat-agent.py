# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import AsyncGenerator

from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.models import CitationMetadata, TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, Server

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

server = Server()


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


@server.agent(
    name="chat_agent",
    description="General purpose chat agent with comprehensive capabilities including research, weather, search, and reasoning. Features dynamic citations and trajectory tracking for transparent interactions.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hello! I'm your Chat Agent - ready to help with any questions or tasks. What can I assist you with today?",
                display_name="Chat Agent",
                tools=[
                    AgentToolInfo(
                        name="Think", 
                        description="Advanced reasoning and analysis for problem-solving, critical thinking, and comprehensive understanding of complex topics and user requests."
                    ),
                    AgentToolInfo(
                        name="Wikipedia", 
                        description="Search comprehensive information from Wikipedia's vast database covering topics like history, science, culture, geography, and general knowledge."
                    ),
                    AgentToolInfo(
                        name="Weather", 
                        description="Get current weather conditions, forecasts, and climate information for any location worldwide to answer weather-related questions."
                    ),
                    AgentToolInfo(
                        name="DuckDuckGo", 
                        description="Search the web for current information, news, facts, and real-time data from across the internet to provide up-to-date answers."
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
        tags=["Chat", "General", "Research", "Assistant"],
        framework="BeeAI",
        programming_language="Python",
        license="Apache 2.0"
    )
)
async def chat_agent(input: list[Message], context: Context) -> AsyncGenerator:
    """
    General purpose chat agent that combines:
    - Dynamic citations from search results
    - Trajectory tracking for transparency
    - Multi-tool integration for comprehensive assistance
    - Always uses Think tool before responding
    """
    
    user_message = input[-1].parts[0].content if input else "Hello"
    
    # Initialize tracking systems
    tool_tracker = TrackedTool("chat_agent")
    trajectory = TrajectoryCapture()
    
    # Show we're starting
    yield MessagePart(metadata=TrajectoryMetadata(
        message=f"💬 Chat Agent processing: '{user_message}'"
    ))
    
    try:
        # Create tracked tools
        tracked_duckduckgo = TrackedDuckDuckGoTool(tool_tracker)
        tracked_wikipedia = TrackedWikipediaTool(tool_tracker)
        tracked_weather = TrackedOpenMeteoTool(tool_tracker)
        
        # Create agent with comprehensive general purpose instructions
        agent = RequirementAgent(
            llm=ChatModel.from_name("ollama:granite3.3:8b"),
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
                    consecutive_allowed=False
                )
            ],
            instructions="""You are a helpful, knowledgeable, and versatile chat assistant. Your goal is to provide accurate, helpful, and comprehensive responses to any user question or request.

            For any query you receive:
            1. First, think about what information would be most helpful
            2. Use Wikipedia for factual information, definitions, historical context, and general knowledge
            3. Use OpenMeteo for weather-related questions and forecasts
            4. Use DuckDuckGo for current information, news, recent events, and real-time data
            
            Always provide:
            - Thoughtful and analytical responses
            - Accurate and well-researched information
            - Conversational and engaging communication
            - Comprehensive assistance while staying focused on user needs
            - Transparent information about your sources
            
            Be helpful, accurate, and engaging while providing the most useful information possible."""
        )
        
        yield MessagePart(metadata=TrajectoryMetadata(
            message="🛠️ Chat Agent initialized with Think, Wikipedia, Weather, and Search tools"
        ))
        
        # Run agent with trajectory middleware
        response = await agent.run(user_message).middleware(
            GlobalTrajectoryMiddleware(target=trajectory, included=[Tool])
        )
        
        response_text = response.answer.text
        
        # Show trajectory steps
        for i, step in enumerate(trajectory.steps):
            if step.strip():
                tool_name = None
                if "ThinkTool" in step:
                    tool_name = "Think"
                elif "WikipediaTool" in step:
                    tool_name = "Wikipedia"  
                elif "OpenMeteoTool" in step:
                    tool_name = "Weather"
                elif "DuckDuckGoSearchTool" in step:
                    tool_name = "DuckDuckGo"
                    
                yield MessagePart(metadata=TrajectoryMetadata(
                    message=f"Step {i+1}: {step}",
                    tool_name=tool_name
                ))
        
        # Generate main response
        yield MessagePart(content=response_text)
        
        # Generate dynamic citations based on tool usage
        citation_count = 0
        for tool_name, tool_output in tool_tracker.results:
            if citation_count >= 10:  # Limit citations to avoid overwhelming
                break
                
            if tool_name == 'Wikipedia' and hasattr(tool_output, 'results') and tool_output.results:
                for result in tool_output.results:
                    if citation_count >= 10:
                        break
                    # Find specific text to cite inline
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
                                citation_count += 1
                                break
                                
            elif tool_name == 'DuckDuckGo' and hasattr(tool_output, 'results') and tool_output.results:
                for result in tool_output.results:
                    if citation_count >= 10:
                        break
                    # Find relevant terms in the response
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
                                citation_count += 1
                                break
                                
            elif tool_name == 'OpenMeteo':
                # Find weather-related words to cite
                weather_words = ["weather", "temperature", "warm", "cool", "forecast", "conditions", "climate", "rain", "sunny", "cloudy", "wind", "humidity"]
                for word in weather_words:
                    if citation_count >= 10:
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
                        citation_count += 1
                        break
        
        yield MessagePart(metadata=TrajectoryMetadata(
            message="✅ Chat Agent completed successfully with citations"
        ))
        
    except Exception as e:
        yield MessagePart(metadata=TrajectoryMetadata(
            message=f"❌ Error: {str(e)}"
        ))
        yield MessagePart(content=f"🚨 Sorry, I encountered an error while processing your request: {str(e)}")


def run():
    """Entry point for the server."""
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()
