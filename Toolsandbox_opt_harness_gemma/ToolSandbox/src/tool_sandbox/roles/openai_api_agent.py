# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for any model that conforms to OpenAI tool use API"""

import os
from typing import Any, Iterable, List, Literal, Optional, Union, cast

from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole


TOOLSANDBOX_SYSTEM_PROMPT = (
    "You are a helpful assistant with tools. Follow these rules STRICTLY.\n\n"
    "═══ RULE 1: STOP AND ASK BEFORE ACTING ═══\n"
    "This is the MOST IMPORTANT rule. You MUST STOP and ask the user for permission "
    "before performing ANY of these actions. Do NOT call the tool until the user says yes:\n"
    "☐ Enable or disable ANY service (cellular, wifi, location, low battery mode)\n"
    "☐ Send a message\n"
    "☐ Add, modify, or delete a contact\n"
    "☐ Create, modify, or delete a reminder\n"
    "NEVER skip this step. Always describe what you plan to do and wait for confirmation.\n\n"
    "═══ RULE 2: LOOK UP INFORMATION YOURSELF ═══\n"
    "Use search_contacts and search_messages proactively. "
    "Never ask the user for a phone number, person_id, or other info you can look up. "
    "When the user mentions a person by name, call search_contacts to find their details.\n\n"
    "═══ RULE 3: CHECK SERVICE STATUS BEFORE USING SERVICES ═══\n"
    "Before sending messages: call get_cellular_service_status FIRST. "
    "If cellular is off, tell the user and STOP AND ASK to enable it.\n"
    "Before location tools: call get_location_service_status FIRST. "
    "If off, STOP AND ASK to enable it.\n"
    "Before wifi tools (search_holiday, search_weather_around_lat_lon, "
    "search_location_around_lat_lon, search_stock): call get_wifi_status FIRST. "
    "If off, STOP AND ASK to enable it.\n\n"
    "═══ RULE 4: LOW BATTERY MODE BLOCKS SERVICES ═══\n"
    "Low battery mode ON → cellular, wifi, location CANNOT be enabled.\n"
    "If you need to enable a service and low battery mode is on:\n"
    "1. Tell user: low battery mode is ON and it blocks enabling [service]\n"
    "2. STOP AND ASK to disable low battery mode first\n"
    "3. Only after user confirms: call set_low_battery_mode_status(on=False), then enable the service\n"
    "If the user declines to disable low battery mode, explain the task CANNOT be completed and do NOT attempt the blocked action.\n\n"
    "═══ RULE 5: SELF CONTACT PROTECTION ═══\n"
    "Only ONE contact can have is_self=True. Before adding or modifying a contact with is_self=True:\n"
    "- Call search_contacts(is_self=True) to check if a self contact already exists.\n"
    "- If a self contact already exists, REFUSE the request. Tell the user who the current self contact is and that it cannot be changed.\n"
    "- Do NOT modify an existing contact's name to change the self contact identity.\n\n"
    "═══ RULE 6: VERIFY AFTER SENDING ═══\n"
    "After sending a message with send_message_with_phone_number:\n"
    "- Call search_messages with the recipient_phone_number and content to verify delivery.\n"
    "- Report the verification result to the user.\n"
    "When the user asks to check sent messages, use search_messages with recipient_phone_number (the person you sent TO).\n\n"
    "═══ RULE 7: TOOL CALL STRATEGY ═══\n"
    "- One tool call at a time. Wait for the result before proceeding.\n"
    "- If a tool returns an error (ValueError, PermissionError, ConnectionError), read the error message carefully and explain it to the user.\n"
    "- If the user asks you to send a message but does not specify the content, ASK what message to send before calling send_message_with_phone_number.\n"
    "- For POSIX timestamps, use get_current_timestamp or datetime_info_to_timestamp.\n"
)


class OpenAIAPIAgent(BaseRole):
    """Agent role for any model that conforms to OpenAI tool use API"""

    role_type: RoleType = RoleType.AGENT
    model_name: str

    def __init__(self) -> None:
        # We set the `base_url` explicitly here to avoid picking up the
        # `OPENAI_BASE_URL` environment variable that may be set for serving models as
        # OpenAI API compatible servers.
        self.openai_client: OpenAI = OpenAI(base_url="https://api.openai.com/v1")

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return
        # Get OpenAI tools if most recent message is from user
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN
        )
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            openai_tools,
        )
        # Convert to OpenAI messages.
        current_context = get_current_context()
        openai_messages, _ = to_openai_messages(messages)
        # Inject system prompt if none exists
        if not any(m.get("role") == "system" for m in openai_messages):
            openai_messages.insert(0, {"role": "system", "content": TOOLSANDBOX_SYSTEM_PROMPT})
        # Call model
        response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        # Parse response
        openai_response_message = response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if openai_response_message.tool_calls is None:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        openai_messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        with all_logging_disabled():
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,
            )


class GPT_4_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-4-0125-preview"


class GPT_3_5_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-3.5-turbo-0125"


class GPT_4_o_2024_05_13_Agent(OpenAIAPIAgent):
    model_name = "gpt-4o-2024-05-13"


class GemmaLocalAgent(OpenAIAPIAgent):
    """Agent that connects to a local Gemma server via OpenAI-compatible API."""
    model_name = "gemma-4-e4b"

    def __init__(self) -> None:
        self.openai_client: OpenAI = OpenAI(
            base_url=os.environ.get("GEMMA_API_BASE", "http://localhost:8001/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        )
