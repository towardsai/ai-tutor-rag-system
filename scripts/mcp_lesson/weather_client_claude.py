#!/usr/bin/env python
import asyncio
import sys
from typing import Any, Optional, List, Dict
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WeatherMCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.anthropic = Anthropic(api_key=api_key)
        
    async def connect_to_server(self, server_script_path: str):
        """Connect to the Weather MCP server
        
        Args:
            server_script_path: Path to the weather_server.py file
        """
        if not server_script_path.endswith('.py'):
            raise ValueError("Weather server script must be a .py file")

        # Set up the server process
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # Initialize the session
        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to Weather Server with tools:", [tool.name for tool in tools])
        
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available weather tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Get available tools from the server
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        claude_response = self.anthropic.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in claude_response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                tool_result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"\n[Using tool: {tool_name} with parameters: {tool_args}]\n")
                final_text.append(f"[Tool result: {tool_result.content}]\n")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": tool_result.content
                        }
                    ]
                })

                # Get next response from Claude with the tool results
                next_response = self.anthropic.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(next_response.content[0].text)

        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nWeather Assistant Started!")
        print("Ask about weather alerts or forecasts, or type 'quit' to exit.")
        print("Example queries:")
        print("  - 'Are there any weather alerts in California?'")
        print("  - 'What's the weather forecast for Washington DC?'")
        print("  - 'Tell me the forecast for latitude 40.7128, longitude -74.0060'")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                print("\nProcessing your query...")
                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python weather_client.py <path_to_weather_server.py>")
        sys.exit(1)

    client = WeatherMCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())