import asyncio
import os
from collections.abc import AsyncGenerator

from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.models import CitationMetadata
from acp_sdk.models.platform import PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, Server

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

server = Server()

@server.agent(
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(ui_type=PlatformUIType.CHAT, display_name="Boston Guide")
        )
    )
)
async def boston_guide_agent(input: list[Message], context: Context) -> AsyncGenerator:
    """Agent that provides Boston travel recommendations with dynamic citations"""
    
    # Custom tool tracking
    tool_calls = []
    
    # Wrap tools to track their usage
    class TrackedDuckDuckGoTool(DuckDuckGoSearchTool):
        async def _run(self, input_data, options, context):
            result = await super()._run(input_data, options, context)
            tool_calls.append(('DuckDuckGo', result))
            return result
    
    class TrackedWikipediaTool(WikipediaTool):
        async def _run(self, input_data, options, context):
            result = await super()._run(input_data, options, context)
            tool_calls.append(('Wikipedia', result))
            return result
    
    class TrackedOpenMeteoTool(OpenMeteoTool):
        async def _run(self, input_data, options, context):
            result = await super()._run(input_data, options, context)
            tool_calls.append(('OpenMeteo', result))
            return result
    
    # Create agent with tracked tools
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[ThinkTool(), TrackedWikipediaTool(), TrackedOpenMeteoTool(), TrackedDuckDuckGoTool()],
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False)
        ],
        instructions="You must use tools to get current information. Use Wikipedia for Boston info, OpenMeteo for weather, and DuckDuckGo for current restaurant recommendations."
    )
    
    # Run agent
    response = await agent.run(str(input[-1]))
    response_text = response.answer.text
    
    # Yield main response
    yield MessagePart(content=response_text)
    
    # Generate citations dynamically based on tool usage
    for tool_name, tool_output in tool_calls:
        if tool_name == 'Wikipedia' and hasattr(tool_output, 'results') and tool_output.results:
            for result in tool_output.results:
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
                            break
        elif tool_name == 'DuckDuckGo' and hasattr(tool_output, 'results') and tool_output.results:
            for result in tool_output.results:
                # Find restaurant names or key terms in the response
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
                            break
        elif tool_name == 'OpenMeteo':
            # Find weather-related words to cite
            weather_words = ["weather", "temperature", "warm", "cool", "forecast", "conditions"]
            for word in weather_words:
                if word in response_text.lower():
                    start_idx = response_text.lower().find(word)
                    yield MessagePart(
                        metadata=CitationMetadata(
                            url="https://open-meteo.com/",
                            title="Open-Meteo Weather API",
                            description="Weather data from Open-Meteo",
                            start_index=start_idx,
                            end_index=start_idx + len(word)
                        )
                    )
                    break

def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()
