import arxiv
import json
import os
from typing import List
from openai import OpenAI
import asyncio
import nest_asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

nest_asyncio.apply()


# Lesson 5: Creating a MCP Client.
class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.openAI = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = "gemini-2.5-pro"
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        """
        Process the query by calling the model and handling tool calls.
        """
        messages = [{"role": "user", "content": query}]

        # Call Model
        completion = self.openAI.chat.completions.create(
            model=self.model, messages=messages, tools=self.available_tools
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
                tool_result = await self.session.call_tool(
                    tool_name, arguments=tool_args
                )
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
                    model=self.model, messages=messages, tools=self.available_tools
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

    async def connect_to_server_and_run(self):
        """
        Connect to the research server and run the chat loop.
        """
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print(
                    "\nConnected to server with tools:", [tool.name for tool in tools]
                )

                self.available_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                    for tool in response.tools
                ]

                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
