from typing import List

from agent_inspect.core.utils import get_config_or_default
from agent_inspect.metrics.constants import (
    INCLUDE_JUDGE_EXPLANATION,
    INCLUDE_PROMPT_SENT_TO_LLMJ,
    TEMPLATE_SUBGOAL,
)
from agent_inspect.metrics.utils.subgoal_validators import SubGoalValidator
from validators.trace_validator_experiment import TraceValidatorExperiment as TraceValidator
from agent_inspect.metrics.validator.subgoal_completion import SubGoalCompletionValidator
from agent_inspect.metrics.validator.templates import (
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
)
from validators.templates_experiment import (
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_NO_STEPS_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_NO_STEPS_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL,
)
from agent_inspect.models.metrics.agent_data_sample import SubGoal
from agent_inspect.models.metrics.agent_trace import TurnTrace
from agent_inspect.models.metrics.validation_result import SubGoalValidationResult


class SubgoalCompletionValidatorExperiment(SubGoalCompletionValidator):
    """
    Variant of SubGoalCompletionValidator that skips the
    TraceValidator.validate_turn_traces_content check in validate_dynamic,
    allowing traces with missing agent_input or agent_response fields.
    """

    async def validate_dynamic(
        self,
        turn_traces: List[TurnTrace],
        sub_goal: SubGoal,
        user_instruction: str | None,
    ) -> SubGoalValidationResult:
        # TraceValidator.validate_turn_traces_content(turn_traces) intentionally omitted
        TraceValidator.validate_agent_trace_if_empty(turn_traces)
        SubGoalValidator.validate_sub_goal(sub_goal)

        include_judge_explanation = get_config_or_default(
            config=self.config, config_key=INCLUDE_JUDGE_EXPLANATION, default=False
        )
        include_entire_prompt_in_validation_result = get_config_or_default(
            config=self.config, config_key=INCLUDE_PROMPT_SENT_TO_LLMJ, default=False
        )

        if user_instruction and user_instruction.strip() != "":
            default_template = (
                DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_NO_STEPS_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL
                if not turn_traces[-1].steps
                else DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL
            )
            template_subgoal = get_config_or_default(
                config=self.config,
                config_key=TEMPLATE_SUBGOAL,
                default=default_template,
            )
            prompt = self.generate_prompt_from_sub_goal_user_task_and_turn_traces(
                sub_goal, user_instruction, turn_traces, template_subgoal
            )
        else:
            default_template = (
                DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_NO_STEPS_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL
                if not turn_traces[-1].steps
                else DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL
            )
            template_subgoal = get_config_or_default(
                config=self.config,
                config_key=TEMPLATE_SUBGOAL,
                default=default_template,
            )
            prompt = self.generate_prompt_from_sub_goal_without_user_task_and_turn_traces(
                sub_goal, turn_traces, template_subgoal
            )

        (
            majority_voted_score,
            judge_explanations,
        ) = await self.get_majority_voted_score_from_judge_responses(prompt=prompt)

        is_completed = True if majority_voted_score == 1 else False
        explanations = []
        if is_completed:
            explanations.append(f'Check: "{sub_goal.details}" has passed successfully.')
        else:
            explanations.append(f'Check: "{sub_goal.details}" has failed.')
        if include_judge_explanation:
            explanations.extend(judge_explanations)
        if include_entire_prompt_in_validation_result:
            return SubGoalValidationResult(
                is_completed=is_completed,
                sub_goal=sub_goal,
                explanations=explanations,
                prompt_sent_to_llmj=prompt,
            )
        return SubGoalValidationResult(
            is_completed=is_completed, sub_goal=sub_goal, explanations=explanations
        )
