"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
    List,
)

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    convert_to_openai_messages,
)
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.graph import (
    START,
    END,
    StateGraph,
    MessagesState,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from langchain_core.messages import merge_message_runs, SystemMessage, HumanMessage
from openai import OpenAIError
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from trustcall import create_extractor

from app.core.config import Environment, settings, Configuration
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.prompts import SYSTEM_PROMPT
from app.schemas import (
    GraphState,
    Message,
)
from app.utils import (
    dump_messages,
    prepare_messages,
)

from app.utils.llm import get_llm
from app.core.prompts.task_prompts import *
from app.schemas.tasks import *
import uuid
import time


class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"])


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use environment-specific LLM model
        llm = get_llm(
            model=settings.LLM_MODEL,
            **self._get_model_kwargs(),
        )
        self.llm = llm[0]
        self.model_name = llm[1]
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self.profile_extractor = create_extractor(
            self.llm,
            tools=[Profile],
            tool_choice="Profile",
        )
        self.spy = Spy()
        self.todo_extractor = create_extractor(
            self.llm, tools=[ToDo], tool_choice="ToDo", enable_inserts=True
        ).with_listeners(on_end=self.spy)

        logger.info("llm_initialized", model=settings.LLM_MODEL, environment=settings.ENVIRONMENT.value)

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get environment-specific model kwargs.

        Returns:
            Dict[str, Any]: Additional model arguments based on environment
        """
        model_kwargs = {}

        # Development - we can use lower speeds for cost savings
        if settings.ENVIRONMENT == Environment.DEVELOPMENT:
            model_kwargs["top_p"] = 0.8

        # Production - use higher quality settings
        elif settings.ENVIRONMENT == Environment.PRODUCTION:
            model_kwargs["top_p"] = 0.95
            model_kwargs["presence_penalty"] = 0.1
            model_kwargs["frequency_penalty"] = 0.1

        return model_kwargs

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                self._connection_pool = AsyncConnectionPool(
                    settings.POSTGRES_URL,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                        "row_factory": dict_row,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def extract_tool_info(self, tool_calls, schema_name="Memory"):
        """Extract information from tool calls for both patches and new memories.

        Args:
            tool_calls: List of tool calls from the model
            schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
        """
        # Initialize list of changes
        changes = []

        for call_group in tool_calls:
            for call in call_group:
                if call["name"] == "PatchDoc":
                    # Check if there are any patches
                    if call["args"]["patches"]:
                        changes.append(
                            {
                                "type": "update",
                                "doc_id": call["args"]["json_doc_id"],
                                "planned_edits": call["args"]["planned_edits"],
                                "value": call["args"]["patches"][0]["value"],
                            }
                        )
                    else:
                        # Handle case where no changes were needed
                        changes.append(
                            {
                                "type": "no_update",
                                "doc_id": call["args"]["json_doc_id"],
                                "planned_edits": call["args"]["planned_edits"],
                            }
                        )
                elif call["name"] == schema_name:
                    changes.append({"type": "new", "value": call["args"]})

        # Format results as a single string
        result_parts = []
        for change in changes:
            if change["type"] == "update":
                result_parts.append(
                    f"Document {change['doc_id']} updated:\n"
                    f"Plan: {change['planned_edits']}\n"
                    f"Added content: {change['value']}"
                )
            elif change["type"] == "no_update":
                result_parts.append(f"Document {change['doc_id']} unchanged:\n" f"{change['planned_edits']}")
            else:
                result_parts.append(f"New {schema_name} created:\n" f"Content: {change['value']}")

        return "\n\n".join(result_parts)

    async def get_memories(self, name: str, user_id: str | int) -> List[Any]:
        """Get memories from the store."""
        try:
            if not self._graph or not self._graph.store:
                raise Exception("Graph or store not initialized")

            namespace = (name, user_id)
            return await self._graph.store.asearch(namespace)
        except Exception as e:
            logger.error(f"Error getting memories for {name}: {str(e)}")
            raise

    async def get_single_memory(self, name: str, memory_name: str, user_id: str | int):
        try:
            if not self._graph or not self._graph.store:
                raise Exception("Graph or checkpointer not initialized")

            namespace = (name, user_id)
            return await self._graph.store.aget(namespace, memory_name)
        except Exception as e:
            err = f"Error while getting memory {memory_name} for store '{name}': {str(e)}"
            logger.error(err)
            raise Exception(err)

    async def put_memories(self, name: str, key: str, value: dict, user_id: str | int):
        try:
            if not self._graph or not self._graph.store:
                raise Exception("Graph or checkpointer not initialized")

            namespace = (name, user_id)
            return await self._graph.store.aput(
                namespace,
                key,
                value,
            )
        except Exception as e:
            err = f"Error while putting memory {key} for store '{name}': {str(e)}"
            logger.error(err)
            raise Exception(err)

    async def task_mAIstro(self, state: MessagesState, config: Configuration):
        """Load memories from the store and use them to personalize the chatbot's response."""
        start_time = time.time()
        try:
            logger.info(f"user_id {config["configurable"]["user_id"]}")
            # Retrieve profile memory from the store
            user_memories = await self.get_memories("profile", config["configurable"]["user_id"])
            if user_memories:
                user_profile = user_memories[0].value
            else:
                user_profile = None

            # Retrieve people memory from the store
            todo_memories = await self.get_memories("todo", config["configurable"]["user_id"])
            todo = "\n".join(f"{mem.value}" for mem in todo_memories)

            # Retrieve custom instructions
            instructions_memories = await self.get_memories("instructions", config["configurable"]["user_id"])
            if instructions_memories:
                instructions = instructions_memories[0].value
            else:
                instructions = ""

            system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)

            # Respond using memory as well as the chat history
            response = await self.llm.bind_tools([UpdateMemory], parallel_tool_calls=False).ainvoke(
                [SystemMessage(content=system_msg)] + state["messages"]
            )

            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error in task_mAIstro: {str(e)}")
            # Return a safe state that allows the graph to continue
            return {"messages": [{"role": "assistant", "content": "I encountered an error. Let me try again."}]}
        finally:
            duration = time.time() - start_time
            logger.info(f"task_mAIstro execution time: {duration:.2f}s")

    async def update_profile(self, state: MessagesState, config: Configuration):
        """Reflect on the chat history and update the memory collection."""

        # Retrieve the most recent memories for context
        existing_items = await self.get_memories("profile", config["configurable"]["user_id"])

        # Format the existing memories for the Trustcall extractor
        tool_name = "Profile"
        existing_memories = (
            [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
            if existing_items
            else None
        )

        # Merge the chat history and the instruction
        TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages = list(
            merge_message_runs(
                messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]
            )
        )

        # Invoke the extractor
        result = await self.profile_extractor.ainvoke({"messages": updated_messages, "existing": existing_memories})

        # Save save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            await self.put_memories(
                "profile", key=rmeta.get("json_doc_id", str(uuid.uuid4())), value=r.model_dump(mode="json")
            )

        tool_calls = state["messages"][-1].tool_calls
        # Return tool message with update verification
        return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_calls[0]["id"]}]}

    async def update_todos(self, state: MessagesState, config: Configuration):
        """Reflect on the chat history and update the memory collection."""

        # Retrieve the most recent memories for context
        existing_items = await self.get_memories("todo", config["configurable"]["user_id"])

        # Format the existing memories for the Trustcall extractor
        tool_name = "ToDo"
        existing_memories = (
            [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
            if existing_items
            else None
        )

        # Merge the chat history and the instruction
        TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages = list(
            merge_message_runs(
                messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]
            )
        )

        # Invoke the extractor
        result = await self.todo_extractor.ainvoke({"messages": updated_messages, "existing": existing_memories})
        logger.info(f"result: {result}")
        # Save save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            await self.put_memories(
                "todo",
                key=rmeta.get("json_doc_id", str(uuid.uuid4())),
                value=r.model_dump(mode="json"),
                user_id=config["configurable"]["user_id"],
            )

        # Respond to the tool call made in task_mAIstro, confirming the update
        tool_calls = state["messages"][-1].tool_calls

        # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
        todo_update_msg = self.extract_tool_info(self.spy.called_tools, tool_name)
        return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id": tool_calls[0]["id"]}]}

    async def update_instructions(self, state: MessagesState, config: Configuration):
        """Reflect on the chat history and update the memory collection."""

        existing_memory = self.get_single_memory(
            "instructions", "user_instructions", config["configurable"]["user_id"]
        )

        # Format the memory in the system prompt
        system_msg = CREATE_INSTRUCTIONS.format(
            current_instructions=existing_memory.value if existing_memory else None
        )
        new_memory = await self.llm.ainvoke(
            [SystemMessage(content=system_msg)]
            + state["messages"][:-1]
            + [HumanMessage(content="Please update the instructions based on the conversation")]
        )

        # Overwrite the existing memory in the store
        key = "user_instructions"
        self.put_memories(
            "instructions",
            key=key,
            value={"memory": new_memory.content},
            user_id=config["configurable"]["user_id"],
        )
        tool_calls = state["messages"][-1].tool_calls
        # Return tool message with update verification
        return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id": tool_calls[0]["id"]}]}

    async def route_message(
        self, state: MessagesState
    ) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
        """Reflect on the memories and chat history to decide whether to update the memory collection."""
        message = state["messages"][-1]
        if len(message.tool_calls) == 0:
            return END
        else:
            tool_call = message.tool_calls[0]
            if tool_call["args"]["update_type"] == "user":
                return "update_profile"
            elif tool_call["args"]["update_type"] == "todo":
                return "update_todos"
            elif tool_call["args"]["update_type"] == "instructions":
                return "update_instructions"
            else:
                raise ValueError

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                builder = StateGraph(MessagesState, config_schema=Configuration)

                # Define the flow of the memory extraction process
                builder.add_node(self.task_mAIstro)
                builder.add_node(self.update_todos)
                builder.add_node(self.update_profile)
                builder.add_node(self.update_instructions)

                # Define the flow
                builder.add_edge(START, "task_mAIstro")
                builder.add_conditional_edges("task_mAIstro", self.route_message)
                builder.add_edge("update_todos", "task_mAIstro")
                builder.add_edge("update_profile", "task_mAIstro")
                builder.add_edge("update_instructions", "task_mAIstro")

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = MemorySaver()  # AsyncPostgresSaver(connection_pool)
                    # await checkpointer.setup()

                    store = AsyncPostgresStore(connection_pool)
                    await store.setup()
                    # Initialize the store
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    store = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = builder.compile(
                    checkpointer=checkpointer,
                    store=store,
                    name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})",
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=store is not None,
                )
                return self._graph
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            list[dict]: The response from the LLM.
        """

        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id, "user_id": "default"},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value,
                    debug=False,
                    user_id=user_id,
                    session_id=session_id,
                )
            ],
        }
        if user_id:
            config["configurable"]["user_id"] = str(user_id)
        logger.info(f"messages: {messages}")
        messages_list = dump_messages(messages)
        logger.info(f"messages_list: {messages_list}")
        try:
            response = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=message["content"]) for message in messages_list]},
                config,
                stream_mode="values",
            )
            return self.__process_messages(response["messages"])
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise e

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        method: Literal["values", "messages"] = "values",
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id, "user_id": "default"},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value, debug=False, user_id=user_id, session_id=session_id
                )
            ],
        }
        if user_id:
            config["configurable"]["user_id"] = str(user_id)
        try:
            messages_list = dump_messages(messages)
            async for chunk in self._graph.astream(
                {"messages": [HumanMessage(content=message["content"]) for message in messages_list]},
                config,
                stream_mode=method,
            ):
                try:
                    if method == "values":
                        logger.info(f"chunk: {chunk}")
                        if isinstance(chunk, dict) and "messages" in chunk:
                            last_message = chunk["messages"][-1]
                            if hasattr(last_message, "content"):
                                yield last_message.content
                            else:
                                logger.warning(f"Message has no content: {last_message}")
                        else:
                            logger.warning(f"Unexpected chunk format: {chunk}")
                    elif method == "messages":
                        token = chunk[0]
                        yield token.content
                except Exception as chunk_error:
                    logger.error("Error processing chunk", error=str(chunk_error), session_id=session_id)
                    continue
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(**message)
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
