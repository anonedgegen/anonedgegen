import uuid
import copy
import json
import time
from pathlib import Path
from tool_sandbox.common.execution_context import RoleType, set_current_context, get_current_context, DatabaseNamespace
from tool_sandbox.common.message_conversion import Message, serialize_to_conversation
from tool_sandbox.cli.utils import resolve_scenarios, AGENT_TYPE_TO_FACTORY, USER_TYPE_TO_FACTORY
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.common.scenario import Scenario, Milestone, Minefield
import polars as pl

# Configure Polars formatting once at module load (process-global, not thread-safe to set per-request)
pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
pl.Config.set_tbl_formatting("ASCII_FULL")

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def start_session(self, agent_type, scenario_name=None):
        session_id = str(uuid.uuid4())

        # If scenario_name is provided, use it; otherwise create a minimal default context with all tools
        if scenario_name:
            scenarios = resolve_scenarios([scenario_name], ToolBackend.DEFAULT)
            scenario = scenarios[scenario_name]
            context = copy.deepcopy(scenario.starting_context)
        else:
            # Use the base scenario's pre-seeded context (Tomas Haake persona, contacts, messages,
            # reminders, device settings) but remove all tool restrictions so every tool is available.
            from scenarios.base_scenarios import named_base_scenarios
            base_scenarios = named_base_scenarios(ToolBackend.DEFAULT)
            scenario = None
            context = copy.deepcopy(base_scenarios["base"].starting_context)
            context.tool_allow_list = None
            context.tool_deny_list = None

        # Create agent, environment, user (fix: use RoleType.USER)
        agent = AGENT_TYPE_TO_FACTORY[agent_type]()
        environment = ExecutionEnvironment()
        user = USER_TYPE_TO_FACTORY[RoleType.USER]() if RoleType.USER in USER_TYPE_TO_FACTORY else None

        set_current_context(context)
        # Handle system messages as in Scenario.play
        sandbox_db = context.get_database(DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False, get_all_history_snapshots=True)
        max_sandbox_message_index = context.max_sandbox_message_index
        for message_index in range(max_sandbox_message_index + 1):
            if (
                sandbox_db["recipient"][message_index] == RoleType.EXECUTION_ENVIRONMENT
                and sandbox_db["sender"][message_index] == RoleType.SYSTEM
            ):
                environment.respond(ending_index=message_index)
        self.sessions[session_id] = {
            "agent_type": agent_type,
            "scenario": scenario,
            "agent": agent,
            "environment": environment,
            "context": context,
        }
        # Do NOT call self.save_trajectory(session_id) here
        return session_id

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def end_session(self, session_id):
        """Remove a session and free its resources."""
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        session.clear()
        return True

    def add_user_message(self, session, message):
        context = session["context"]
        set_current_context(context)
        msg = Message(sender=RoleType.USER, recipient=RoleType.AGENT, content=message)
        BaseRole.add_messages([msg])
        
        # Update session context
        session["context"] = context

    def _wrap_agent_for_usage_tracking(self, agent):
        """Wrap the agent's model_inference to capture token usage from each call."""
        if hasattr(agent, '_usage_wrapped'):
            return
        agent._usage_tracking = []
        original_model_inference = agent.model_inference

        def tracked_model_inference(*args, **kwargs):
            response = original_model_inference(*args, **kwargs)
            if hasattr(response, 'usage') and response.usage is not None:
                agent._usage_tracking.append({
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                })
            return response

        agent.model_inference = tracked_model_inference
        agent._usage_wrapped = True

    def _collect_usage(self, agent):
        """Collect and reset accumulated usage from the agent."""
        if not hasattr(agent, '_usage_tracking') or not agent._usage_tracking:
            return None
        calls = agent._usage_tracking
        aggregated = {
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "total_tokens": sum(c["total_tokens"] for c in calls),
            "num_calls": len(calls),
            "per_call": calls,
        }
        agent._usage_tracking = []
        return aggregated

    def agent_respond(self, session, session_id=None):
        context = session["context"]
        set_current_context(context)
        agent = session["agent"]
        environment = session["environment"]

        # Wrap agent to track usage
        self._wrap_agent_for_usage_tracking(agent)

        # Turn-taking loop as in Scenario.play - exactly match the original implementation
        max_messages = 30  # Default from scenario
        initial_max_index = context.max_sandbox_message_index

        while True:
            sandbox_db = context.get_database("SANDBOX", get_all_history_snapshots=True, drop_sandbox_message_index=False)

            # Check termination conditions
            conversation_active = sandbox_db["conversation_active"][-1]
            current_message_index = sandbox_db["sandbox_message_index"][-1]

            if not conversation_active or current_message_index >= max_messages + initial_max_index:
                break

            last_msg_recipient = sandbox_db["recipient"][-1]

            if last_msg_recipient == RoleType.AGENT:
                # print(f"[DEBUG] Making agent call... (message index: {current_message_index})")
                try:
                    agent.respond()
                    # print(f"[DEBUG] Agent call completed")
                except Exception as e:
                    # print(f"[ERROR] Agent call failed: {e}")
                    break
            elif last_msg_recipient == RoleType.EXECUTION_ENVIRONMENT:
                # print(f"[DEBUG] Making execution environment call...")
                environment.respond()
                # print(f"[DEBUG] Execution environment call completed")
            elif last_msg_recipient == RoleType.USER:
                # Update session context before saving trajectory
                session["context"] = get_current_context()
                # Save trajectory after each user turn
                if session_id:
                    self.save_trajectory(session_id)
                # Return the last agent message with usage
                usage = self._collect_usage(agent)
                msgs = sandbox_db.to_dicts()
                for msg in reversed(msgs):
                    if msg["sender"] == RoleType.AGENT:
                        return msg["content"], usage
                return None, usage
            else:
                break

            # Refresh the database for the next iteration (exactly like original)
            context = get_current_context()
            set_current_context(context)

        # Update session context before returning
        session["context"] = get_current_context()

        # If we exit the loop, return the last agent message with usage
        usage = self._collect_usage(agent)
        sandbox_db = context.get_database("SANDBOX", get_all_history_snapshots=True, drop_sandbox_message_index=False)
        msgs = sandbox_db.to_dicts()
        for msg in reversed(msgs):
            if msg["sender"] == RoleType.AGENT:
                return msg["content"], usage
        return None, usage

    def save_trajectory(self, session_id):
        session = self.sessions[session_id]
        context = session["context"]
        scenario = session["scenario"]
        output_dir = Path("data") / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Polars to show full content without truncation
        pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
        pl.Config.set_tbl_formatting("ASCII_FULL")

        pretty_print_str = (
            "Note that User Simulator few shot messages have been omitted\n"
            + str(
                context.get_database(
                    DatabaseNamespace.SANDBOX,
                    get_all_history_snapshots=True,
                    drop_sandbox_message_index=False,
                )
                .filter(
                    (pl.col("visible_to") != [RoleType.USER])
                    | (pl.col("visible_to").is_null())
                )
                .drop([
                    "openai_tool_call_id",
                    "conversation_active",
                ])
            )
        )
        with open(output_dir / "pretty_print.txt", "w", encoding="utf-8") as f:
            f.write(pretty_print_str)
        # Execution context
        with open(output_dir / "execution_context.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(context.to_dict(serialize_console=False), ensure_ascii=False, indent=4))
        # Conversation
        milestones = []
        minefields = []
        if scenario and hasattr(scenario, 'evaluation'):
            milestone_matcher = getattr(scenario.evaluation, "milestone_matcher", None)
            minefield_matcher = getattr(scenario.evaluation, "minefield_matcher", None)
            if milestone_matcher:
                milestones = milestone_matcher.milestones
            if minefield_matcher:
                minefields = minefield_matcher.milestones

        conversation = serialize_to_conversation(
            context,
            None,
            milestones,
            minefields,
        )
        # Group messages into turns: each turn starts with a user message
        turns = []
        current_turn = []
        for msg in conversation:
            if msg.get("role") == "system":
                continue
            if msg.get("role") == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)
        with open(output_dir / "conversation.json", "w", encoding="utf-8") as f:
            json.dump(turns, f, indent=2, ensure_ascii=False)
    
    def get_trajectory(self, session_id):
        """Get the saved trajectory for debugging purposes"""
        output_dir = Path("data") / session_id
        conversation_file = output_dir / "conversation.json"
        if conversation_file.exists():
            with open(conversation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []