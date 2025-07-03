import httpx
from collections.abc import AsyncGenerator
from acp_sdk import MessagePart, Metadata, Annotations
from acp_sdk.models import Message
from acp_sdk.models.platform import PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.tools.think import ThinkTool

server = Server()

@server.agent(
    name="python_doc_generator",
    description="Generate documentation for Python files using BeeAI Framework RequirementAgent",
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.HANDSOFF,
                user_greeting="📚 Upload a Python file and I'll generate comprehensive documentation!",
                display_name="Python Documentation Generator"
            )
        )
    )
)
async def python_doc_generator(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Generate documentation for Python files using BeeAI Framework RequirementAgent"""
    
    # Check for uploaded files
    for part in input[-1].parts:
        if part.content_url:
            yield MessagePart(content="📁 Processing uploaded file...")
            
            try:
                # Download file content
                async with httpx.AsyncClient() as client:
                    response = await client.get(str(part.content_url))
                    code = response.content.decode('utf-8')
                    
                    # Basic Python validation
                    if not any(indicator in code for indicator in ['def ', 'class ', 'import ', 'print(']):
                        yield MessagePart(content="⚠️ This doesn't appear to be a Python file.")
                        return
                    
                    yield MessagePart(content="🤖 Creating documentation with RequirementAgent...")
                    
                    # Create RequirementAgent (uses platform LLM config)
                    agent = RequirementAgent(
                        llm=ChatModel.from_name("ollama:granite3.3:8b"),
                        tools=[ThinkTool()],
                        requirements=[
                            ConditionalRequirement(ThinkTool, force_at_step=1, consecutive_allowed=False)
                        ]
                    )
                    
                    # Let the LLM generate comprehensive documentation
                    prompt = f"""Please analyze this Python code and generate comprehensive documentation that explains:

1. What this code does and its main purpose
2. How the code works and its structure  
3. Key functions, classes, and components
4. Dependencies and imports used
5. How to use this code
6. Any important technical details

Here's the Python code:

```python
{code}
```

Please provide clear, detailed documentation that helps developers understand this code."""
                    
                    # Run the agent
                    response = await agent.run(prompt)
                    
                    # Output the documentation
                    yield MessagePart(content=f"\n# 📖 Python Code Documentation\n\n{response.answer.text}")
                    
            except Exception as e:
                yield MessagePart(content=f"❌ Error: {str(e)}")
                return
    
    # No files uploaded
    yield MessagePart(content="Please upload a Python file to generate documentation!")

def run():
    """Entry point for the server."""
    server.run()

if __name__ == "__main__":
    run()
