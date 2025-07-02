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
    description="Showcases the latest trajectory feature on the BeeAI Platform using the RequirementAgent from the BeeAI Framework.",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
                user_greeting="Hi! I can help you search Wikipedia, check the weather, or think through problems. What would you like to do?",
                display_name="Trajectory Agent",
                tools=[
                    AgentToolInfo(
                        name="Wikipedia", 
                        description="Search and retrieve factual information from Wikipedia, the world’s largest free encyclopedia. This tool can be used to answer general knowledge questions, summarize topics, and provide background context across a wide range of domains including history, science, geography, and more."
                    ),
                    AgentToolInfo(
                        name="Weather", 
                        description="Access current weather conditions, forecasts, and meteorological data for any location worldwide. Useful for planning, travel, event coordination, and decision-making based on weather patterns like temperature, precipitation, wind speed, and alerts."
                    ),
                    AgentToolInfo(
                        name="Think", 
                        description="Engage in advanced reasoning, structured analysis, and problem-solving. This tool enables the agent to break down complex scenarios, weigh options, infer implications, generate hypotheses, and develop multi-step solutions or strategic recommendations."
                    ),
                ]
            )
        ),
        author={
            "name": "Jenna Winkler",
            "email": "test@example.com",
            "url": "https://johndoe.dev"
        },
        recommended_models=[
            "granite3.3:8b-beeai"
        ],
        tags=["Research"],
        framework="BeeAI",
        programming_language="Python",
        license="Apache 2.0"
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
