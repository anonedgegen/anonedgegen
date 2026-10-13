from typing import List

from agent_inspect.exception import InvalidInputValueError
from agent_inspect.exception.error_codes import ErrorCode
from agent_inspect.models.metrics.agent_trace import TurnTrace


class TraceValidatorExperiment:
    @staticmethod
    def validate_agent_trace_if_empty(agent_turn_traces: List[TurnTrace]) -> None:
        if not agent_turn_traces:
            raise InvalidInputValueError(
                internal_code=ErrorCode.MISSING_VALUE.value,
                message="Agent turn traces list is empty or None.",
            )
