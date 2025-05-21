#!/usr/bin/env python
import asyncio
import sys
from typing import Any, Optional, List, Dict
from contextlib import AsyncExitStack
import os
from dotenv import load_dotenv
import json 

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI 

load_dotenv()

class WeatherMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_provider_name = "OpenAI"

        # --- OPENAI API CLIENT INITIALIZATION ---
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY not found. OpenAI client might fail or use a default configuration if available.", file=sys.stderr)
        self.openai_llm_client = OpenAI(api_key=openai_api_key)
        

    async def connect_to_server(self, server_script_path: str):
        """Connect to the Weather MCP server"""
        if not server_script_path.endswith('.py'):
            raise ValueError("Weather server script must be a .py file")

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script_path],
            env=os.environ.copy()
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print(f"\nConnected to Weather Server. LLM Provider: {self.llm_provider_name}. Available tools: {[tool.name for tool in tools]}")

    async def process_query(self, query: str) -> str:
        """Process a query using the OpenAI LLM and available weather tools"""
        if not self.openai_llm_client:
            return "Error: OpenAI client not initialized."

        final_response_parts = []

        mcp_tools_response = await self.session.list_tools()
        tool_descriptions_for_prompt = []
        if mcp_tools_response and mcp_tools_response.tools:
            for tool in mcp_tools_response.tools:
                schema_str = "No input schema defined"
                if tool.inputSchema and isinstance(tool.inputSchema, dict):
                    schema_str = json.dumps(tool.inputSchema)
                elif tool.inputSchema:
                    schema_str = str(tool.inputSchema)
                tool_descriptions_for_prompt.append(f"  - Name: {tool.name}\n    Description: {tool.description}\n    Input Schema (parameters as JSON object): {schema_str}")

        tools_info_prompt_section = "You are a helpful weather assistant. You have access to the following tools to help answer weather-related questions:\n" + "\n".join(tool_descriptions_for_prompt)
        tools_info_prompt_section += (
            "\n\nBased on the user's query, decide if a tool is needed. "
            "If a tool can help, respond ONLY with the tool call formatted EXACTLY like this (including the 'TOOL_CALL:' prefix and JSON structure, ensure parameters is a JSON object): "
            "TOOL_CALL: {\"tool_name\": \"<tool_name_here>\", \"parameters\": {<parameters_here_as_json_object>}}. "
            "If you don't need a tool, or if the query is not weather-related and cannot be answered by tools, just answer the query directly based on your general knowledge. "
            "If a tool requires latitude and longitude and they are not provided, you can ask the user for them, or state you need them to proceed with the tool."
        )

        openai_first_pass_messages = [
            {"role": "system", "content": tools_info_prompt_section},
            {"role": "user", "content": f"User Query: {query}"}
        ]

        try:
            first_pass_completion = self.openai_llm_client.chat.completions.create(
                model="gpt-4o", 
                messages=openai_first_pass_messages,
                max_tokens=500 
            )
            first_pass_text = first_pass_completion.choices[0].message.content.strip()
            

        except Exception as e:
            print(f"Error calling OpenAI API (first pass): {e}", file=sys.stderr)
            return f"Error processing query with OpenAI (first pass): {e}"

        if first_pass_text.startswith("TOOL_CALL:"):
            try:
                tool_call_json_str = first_pass_text.replace("TOOL_CALL:", "").strip()
                tool_call_data = json.loads(tool_call_json_str)
                tool_name = tool_call_data.get("tool_name")
                tool_parameters = tool_call_data.get("parameters", {})

                if not tool_name:
                    raise ValueError("Invalid TOOL_CALL format from LLM: 'tool_name' missing.")
                if not isinstance(tool_parameters, dict):
                     raise ValueError(f"Invalid TOOL_CALL format from LLM: 'parameters' is not a JSON object. Got: {tool_parameters}")

                final_response_parts.append(f"[OpenAI decided to use tool: {tool_name} with parameters: {json.dumps(tool_parameters)}]")

                mcp_tool_result = await self.session.call_tool(tool_name, tool_parameters)
                tool_result_content = str(mcp_tool_result.content)
                final_response_parts.append(f"[Tool {tool_name} result: {tool_result_content}]")

                openai_second_pass_messages = [
                    {"role": "system", "content": "You are a helpful weather assistant. You have received the result from a tool call. Use this information to formulate a comprehensive, natural language answer to the user's original query."},
                    {"role": "user", "content": f"Original query: '{query}'\nTool used: '{tool_name}'\nTool parameters: {json.dumps(tool_parameters)}\nTool result: '{tool_result_content}'\n\nPlease provide the final answer to the user."},
                ]
               
                second_pass_completion = self.openai_llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=openai_second_pass_messages,
                    max_tokens=1500
                )
                final_answer_text = second_pass_completion.choices[0].message.content.strip()
                final_response_parts.append(f"\n{final_answer_text}")

            except json.JSONDecodeError as jde:
                final_response_parts.append(f"\n[LLM provided an invalid JSON for TOOL_CALL: {tool_call_json_str}. Error: {jde}]")
                final_response_parts.append("I tried to use a tool but received an improperly formatted instruction. Could you try rephrasing?")
            except ValueError as ve:
                final_response_parts.append(f"\n[Error parsing TOOL_CALL: {ve}]")
                final_response_parts.append("I tried to use a tool but encountered an issue with its specification. Could you try rephrasing?")
            except Exception as e:
                final_response_parts.append(f"\n[Error executing tool or during second OpenAI call: {type(e).__name__} - {e}]")
                final_response_parts.append("Sorry, I encountered an error while trying to use a tool to answer your question.")
        else: 
            final_response_parts.append(first_pass_text)

        return "\n".join(filter(None, final_response_parts))

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nWeather Assistant Started!")
        print(f"Using LLM Provider: {self.llm_provider_name}")
        print("Ask about weather alerts or forecasts, or type 'quit' to exit.")
        print("Example queries:")
        print("  - 'Are there any weather alerts in California?'")
        print("  - 'What's the weather forecast for Washington DC?' (May require lat/lon for forecast tool)")
        print("  - 'Tell me the forecast for latitude 40.7128, longitude -74.0060'")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
                if query.lower() == 'quit':
                    break

                print("\nProcessing your query...")
                response = await self.process_query(query)
                print("\n" + response)

            except KeyboardInterrupt:
                print("\nExiting chat loop...")
                break
            except Exception as e:
                print(f"\nAn error occurred in the chat loop: {type(e).__name__} - {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        await self.exit_stack.aclose()
        print("Cleanup complete.")

async def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.executable} {os.path.basename(__file__)} <path_to_weather_server.py>")
        sys.exit(1)

    server_path = sys.argv[1]
    if not os.path.exists(server_path):
        print(f"Error: Server script not found at {server_path}")
        sys.exit(1)
    if not os.path.isfile(server_path) or not server_path.endswith(".py"):
        print(f"Error: Server path {server_path} must be a Python file.")
        sys.exit(1)

    client = WeatherMCPClient()
    try:
        await client.connect_to_server(os.path.abspath(server_path))
        await client.chat_loop()
    except Exception as e:
        print(f"An unhandled error occurred: {type(e).__name__} - {str(e)}", file=sys.stderr)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Critical error during startup/shutdown: {type(e).__name__} - {e}", file=sys.stderr)