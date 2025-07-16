import arxiv
import json
import os
from openai import OpenAI
import asyncio
import nest_asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from dataclasses import dataclass
from contextlib import AsyncExitStack

nest_asyncio.apply()

# Lesson 6: Connecting the MCP Chatbot to Reference Servers


@dataclass
class ToolDefinition:
    """
    Represents the definition of a tool that can be called by the AI model.
    """

    name: str
    description: str
    input_schema: dict


class MCP_ChatBot:
    """
    A chatbot that connects to multiple MCP servers, retrieves available tools,
    and processes user queries by calling the model and executing tool calls.
    """

    def __init__(self):
        """
        Initialize the MCP_ChatBot instance.
        """
        # Initialize session and tools
        self.sessions: list[ClientSession] = []
        self.available_tools: list[ToolDefinition] = []
        self.tool_to_session: dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()

        # Initialize LLM client
        self.openAI = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = "gemini-2.5-pro"

    async def connect_to_servers(self):
        """
        Connect to all servers defined in `server_config.json`.
        """
        try:
            with open("server_config.json", "r") as f:
                server_config = json.load(f)

            servers = server_config.get("mcpServers", {})

            for server_name, config in servers.items():
                await self.connect_to_server(server_name, config)
                print(f"Connected to server: {server_name}")
        except Exception as e:
            print(f"Error connecting to servers: {str(e)}")

    async def connect_to_server(self, server_name: str, server_config: dict):
        """
        Connect to a specific server using the provided configuration.
        """
        try:
            # Create server parameters & establish stdio transport
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport

            # Create a new ClientSession for this server
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print(
                f"Connected to server {server_name} with tools: {[tool.name for tool in tools]}"
            )

            # Store tools and their definitions
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.inputSchema,
                    )
                )

        except Exception as e:
            print(f"Error connecting to server {server_name}: {str(e)}")

    async def process_query(self, query):
        """
        Process the query by calling the model and handling tool calls.
        """
        messages = [{"role": "user", "content": query}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.available_tools
        ]

        # Call Model
        completion = self.openAI.chat.completions.create(
            model=self.model, messages=messages, tools=tools
        )
        message = completion.choices[0].message
        # print(f"Message: {message}")

        process_query = True
        while process_query:
            # Append the assistant message
            messages.append(message)

            # Check if the message contains tool calls
            if message.tool_calls:
                tool_call = message.tool_calls[0]

                tool_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"Calling tool {tool_name} with args {tool_args}")
                # Execute the tool, tool invocation through the client session
                session = self.tool_to_session.get(tool_name)
                tool_result = await session.call_tool(tool_name, arguments=tool_args)
                # print(f"Tool result: {tool_result}")

                # Append the tool result to the messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result.content,
                    }
                )

                # Call the model again with the updated messages
                completion = self.openAI.chat.completions.create(
                    model=self.model, messages=messages, tools=tools
                )
                message = completion.choices[0].message
                # print(f"Message: {message}")
            else:
                # No tool calls, we can stop processing
                print(message.content)
                process_query = False

    async def chat_loop(self):
        """
        Run an interactive chat loop
        """
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break

                await self.process_query(query)
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """
        Cleanup resources and close all sessions.
        """
        await self.exit_stack.aclose()
        print("Cleanup completed. All sessions closed.")


async def main():
    """
    Main function to run the MCP ChatBot.
    """
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
