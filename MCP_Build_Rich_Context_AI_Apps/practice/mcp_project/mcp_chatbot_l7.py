import arxiv
import json
import os
from openai import OpenAI
import asyncio
import nest_asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

nest_asyncio.apply()

# Lesson 7: Adding Prompt & Resource Features


class MCP_ChatBot:
    """
    A chatbot that connects to multiple MCP servers, retrieves available tools,
    and processes user queries by calling the model and executing tool calls.
    """

    def __init__(self):
        """
        Initialize the MCP_ChatBot instance.
        """
        # Initialize session and other primitives
        self.sessions: list[ClientSession] = []
        self.available_tools: list[types.Tool] = []
        self.available_prompts: dict[str, types.Prompt] = {}
        self.available_resources: dict[str, types.Resource] = {}

        self.tool_to_session: dict[str, ClientSession] = {}
        self.prompt_to_session: dict[str, ClientSession] = {}
        self.resource_to_session: dict[str, ClientSession] = {}

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
            print(f"Connected to server {server_name}")

            try:
                # List & Store available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(
                    f"Tools available on {server_name}: {[tool.name for tool in tools]}"
                )

                for tool in tools:
                    self.available_tools.append(tool)
                    self.tool_to_session[tool.name] = session

                # List & Store available prompts
                prompts_result = await session.list_prompts()
                if prompts_result and prompts_result.prompts:
                    prompts = prompts_result.prompts
                    print(
                        f"Prompts available on {server_name}: {[prompt.name for prompt in prompts]}"
                    )

                    for prompt in prompts:
                        self.available_prompts[prompt.name] = prompt
                        self.prompt_to_session[prompt.name] = session

                # List & Store available resources
                resources_result = await session.list_resources()
                if resources_result and resources_result.resources:
                    resources = resources_result.resources
                    print(
                        f"Resources available on {server_name}: {[resource.name for resource in resources]}"
                    )

                    for resource in resources:
                        self.available_resources[resource.name] = resource
                        self.resource_to_session[resource.name] = session
            except Exception as e:
                print(
                    f"Error listing tools/prompts/resources on {server_name}: {str(e)}"
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
                    "parameters": tool.inputSchema,
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

    async def list_resources(self):
        """
        List all available resources.
        """
        if not self.available_resources:
            print("No resources available.")
            return

        print("\nAvailable Resources:")
        for name, resource in self.available_resources.items():
            print(f"- {name}: {resource.description}")

    async def get_resource(self, name: str):
        """
        Retrieve a specific resource by name.
        """
        resource = self.available_resources.get(name)
        session = self.resource_to_session.get(name)

        if not resource or not session:
            print(f"Resource {name} not found.")

        try:
            result = await session.read_resource(uri=resource.uri)
            if result and result.contents:
                print(f"\nRetrieved resource {name}")
                print("Content:")
                print(result.contents[0].text)
        except Exception as e:
            print(f"Error retrieving resource {name}: {str(e)}")

    async def list_prompts(self):
        """
        List all available prompts.
        """
        if not self.available_prompts:
            print("No prompts available.")
            return

        print("\nAvailable Prompts:")
        for name, prompt in self.available_prompts.items():
            print(f"- {name}: {prompt.description}")
            if prompt.arguments:
                print(f"  Arguments: ")
                for arg in prompt.arguments:
                    print(f"  - {arg.name}: {arg.description}")

    async def execute_prompt(self, name: str, args: dict[str, str] | None):
        """Execute a prompt with the given arguments."""
        session = self.prompt_to_session.get(name)
        if not session:
            print(f"Session for prompt {name} not found.")
            return

        try:
            result = await session.get_prompt(name, arguments=args)
            if result and result.messages and result.messages[0].content:
                content = result.messages[0].content

                # Extract text from content (handles different formats)
                if isinstance(content, types.TextContent):
                    text = content.text
                else:
                    text = None

                if text:
                    print(f"\nExecuting prompt '{name}'...")
                    await self.process_query(text)

        except Exception as e:
            print(f"Error executing prompt {name}: {str(e)}")
            return

    async def chat_loop(self):
        """
        Run an interactive chat loop
        """
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @resources to list available resources")
        print("Use @resource <name> to retrieve a specific resource")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue

                # Check - 'Quit' command
                if query.lower() == "quit":
                    break

                # Check - '@resources' command
                elif query.lower() == "@resources":
                    await self.list_resources()
                    continue

                # Check - '@resource <name>' command
                elif query.lower().startswith("@resource "):
                    resource_name = query[len("@resource ") :].strip()
                    await self.get_resource(resource_name)
                    continue

                # Check - '/prompts' command
                elif query.lower() == "/prompts":
                    await self.list_prompts()
                    continue

                # Check - '/prompt <name> <args>' command
                elif query.lower().startswith("/prompt "):
                    parts = query[len("/prompt ") :].strip().split(" ")
                    prompt_name = parts[0]
                    prompt_args = {}
                    if len(parts) > 1:
                        for arg in parts[1:]:
                            key, value = arg.split("=")
                            prompt_args[key] = value

                    await self.execute_prompt(prompt_name, prompt_args)
                    continue

                # Process the query
                else:
                    await self.process_query(query)

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
