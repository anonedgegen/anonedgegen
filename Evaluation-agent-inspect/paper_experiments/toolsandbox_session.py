import sys

sys.path.append("../")
# TODO: Set the file paths correctly by setting in the __init__ folder and not using sys path

from typing import Any, Dict, List
import os
import requests
import time

from .session import BaseSession, SessionMessageResponse


class ToolsandboxSession(BaseSession):
    """Tool sandbox session implementation."""

    def __init__(self, scenario_name: str, agent_type: str = "GPT_4_1"):
        """Initialize ToolSandbox session with scenario-specific configuration."""
        self.scenario_name = scenario_name
        self.turn_usages: List[Dict[str, Any]] = []
        super().__init__(scenario_name, agent_type)

    def get_config(self) -> Dict[str, Any]:
        """Return ToolSandbox configuration."""
        return {
            "base_url": os.environ.get("SERVER_ENDPOINT", "http://localhost:8000"),
            "endpoints": {
                "start": "/start_session",
                "message": "/message",
                "trajectory": "/trajectory/{session_id}",
                "end": "/end_session/{session_id}",
            },
        }

    def send_message(self, message: str, retries: int = 3) -> Dict[str, Any]:
        """Send a message to the current session."""
        self._ensure_session_active()

        url = f"{self.base_url}{self.endpoints['message'].format(session_id=self.session_id)}"
        payload = self.build_message_payload(message)

        self.logger.info(f"Sending message to session {self.session_id}: {message}")
        for attempt in range(retries):
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                content = data["content"]
                self.turn_usages.append(data.get("usage"))
                return SessionMessageResponse(response_message=content)
            else:
                self.logger.error(
                    f"Attempt {attempt + 1} failed: {response.status_code} - {response.text}"
                )
                self.logger.info("Retrying...")
                time.sleep(3)

        error_msg = f"Failed to send message after {retries} attempts."
        self.logger.error(error_msg)
        raise Exception(error_msg)

    def build_start_payload(self) -> Dict[str, Any]:
        """Create payload for starting a ToolSandbox session.

        Pass the scenario name directly to the server so named ToolSandbox
        scenarios load the correct pre-seeded context. Map 'synthetic' (used
        for synthesized tc_* test cases) to None so the server uses base data.
        """
        scenario = None if self.scenario_name == "synthetic" else self.scenario_name
        return {
            "scenario": scenario,
            "agent_type": self.agent_type,
        }

    def process_trajectory(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ToolSandbox trajectory response data.

        Injects accumulated usage data into assistant messages so that
        extract_trajectory_stats can report token counts.
        """
        usage_idx = 0
        for turn in response_data:
            if not isinstance(turn, list):
                continue
            injected = False
            for msg in turn:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and not injected:
                    if usage_idx < len(self.turn_usages) and self.turn_usages[usage_idx]:
                        msg["usage"] = self.turn_usages[usage_idx]
                        injected = True
            if injected:
                usage_idx += 1
        return response_data

    def build_message_payload(self, message: str) -> Dict[str, Any]:
        """Create payload for sending a message in ToolSandbox."""
        return {
            "scenario": self.scenario_name,
            "session_id": self.session_id,
            "message": message,
            "agent_type": self.agent_type,
        }

    def get_agent_desc(self) -> str:
        """Get agent description for ToolSandbox scenario."""
        self.logger.info(f"Retrieving agent description for scenario {self.scenario_name}")

        # Add domain-specific logic here as needed
        # Example:
        # if self.scenario_name == "search_message_with_recency_latest_multiple_user_turn":
        #     return "Search Message Agent with access to get_timestamp function..."

        return ""

    def end(self) -> None:
        """End the session and free server-side resources."""
        if self.session_id:
            url = f"{self.base_url}{self.endpoints['end'].format(session_id=self.session_id)}"
            try:
                requests.post(url)
            except Exception as e:
                self.logger.warning(f"Failed to end session {self.session_id}: {e}")
        self.session_id = None
