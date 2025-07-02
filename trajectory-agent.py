# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import AsyncGenerator

from acp_sdk import Annotations, MessagePart, Metadata
from acp_sdk.models import Message
from acp_sdk.models.models import TrajectoryMetadata
from acp_sdk.models.platform import AgentToolInfo, PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, Server

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

server = Server()


class TrajectoryCapture:
    def __init__(self):
        self.steps = []
    
    def write(self, message: str) -> int:
        self.steps.append(message.strip())
        return len(message)


@server.agent(
    name="trajectory_agent",
    description="Demonstrates the latest trajectory feature on the BeeAI Platform using the RequirementAgent from the BeeAI Framework.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi! I can search Wikipedia, get weather data, and think through problems. What would you like to know?",
                display_name="Trajectory Agent",
                tools=[
                    AgentToolInfo(name="Wikipedia", description="Search and retrieve information from Wikipedia articles"),
                    AgentToolInfo(name="Weather", description="Get current weather conditions and forecasts by location"),
                    AgentToolInfo(name="Think", description="Perform reasoning, analysis, and structured thought processing"),
                ]
            )
        )
    )
)
async def trajectory_agent(input: list[Message], context: Context) -> AsyncGenerator:
    user_message = input[-1].parts[0].content if input else "Hello"
    
    # Show we're starting
    yield MessagePart(metadata=TrajectoryMetadata(
        message=f"Processing: '{user_message}' with Trajectory Agent..."
    ))
    
    try:
        # Set up trajectory capture
        trajectory = TrajectoryCapture()
        
        # Create trajectory_agent with tools and requirements  
        agent = RequirementAgent(
            llm=ChatModel.from_name("ollama:granite3.3:8b"),
            tools=[ThinkTool(), WikipediaTool(), OpenMeteoTool()],
            requirements=[
                ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False)
            ],
        )
        
        yield MessagePart(metadata=TrajectoryMetadata(
            message="Trajectory Agent initialized with Think, Wikipedia, and Weather tools"
        ))
        
        # Run agent with trajectory middleware
        response = await agent.run(user_message).middleware(
            GlobalTrajectoryMiddleware(target=trajectory, included=[Tool])
        )
        
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
                    
                yield MessagePart(metadata=TrajectoryMetadata(
                    message=f"Step {i+1}: {step}",
                    tool_name=tool_name
                ))
        
        yield MessagePart(metadata=TrajectoryMetadata(
            message="Trajectory Agent completed successfully"
        ))
        
        # Final response
        yield MessagePart(content=response.answer.text)
        
    except Exception as e:
        yield MessagePart(metadata=TrajectoryMetadata(
            message=f"Error: {str(e)}"
        ))
        yield MessagePart(content=f"Sorry, I encountered an error: {str(e)}")


def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    run()