import argparse
import asyncio
import json
import logging
import pickle
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
from uuid import uuid4

from agent_inspect.metrics.adapters import (
    BaseAdapter,
    Tau2BenchAdapter,
    ToolsandboxAdapter,
)
from agent_inspect.metrics.constants import (
    K_VALUE,
    NO_OF_TRIALS,
    INCLUDE_VALIDATION_RESULTS,
    INCLUDE_JUDGE_EXPLANATION,
    OPTIMIZE_JUDGE_TRIALS,
    TEMPLATE_SUBGOAL,
    INCLUDE_PROMPT_SENT_TO_LLMJ,
)
from agent_inspect.metrics.scorer import ProgressScore
from agent_inspect.metrics.scorer.templates import (
    DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_TEMPLATE_ONE_SUBGOAL,
)
from agent_inspect.metrics.multi_samples import PassAtK, PassHatK
from agent_inspect.metrics.scorer import SuccessBasedMetric
from validators.subgoal_completion_experiment import SubgoalCompletionValidatorExperiment
from agent_inspect.tools import UnsupervisedSubgoalErrorAnalysis
from agent_inspect.models.metrics import NumericalScore
from agent_inspect.models.tools import (
    SubgoalErrorAnalysisDataSample,
    SubgoalErrorAnalysisResult,
)
from agent_inspect.models.user_proxy import (
    ChatHistory,
    ConversationTurn,
    ResponseFromAgent,
    TerminatingCondition,
)
from agent_inspect.user_proxy.constants import USE_EXPERT_AGENT
from agent_inspect.user_proxy import UserProxyAgent
from paper_experiments.azure_openai_client import AzureOpenAIClient

from paper_experiments.session import BaseSession
from paper_experiments.tau2bench_session import Tau2BenchSession
from paper_experiments.toolsandbox_session import ToolsandboxSession
from agent_inspect.models.metrics import EvaluationSample, SubGoal


logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """
    Configuration parameters for running agent evaluation experiments.

    This class defines all the settings needed to evaluate an agent's performance
    on a set of tasks, including the agent type, conversation limits, user proxy
    behavior, and parallelization settings.

    Attributes:
        agent_type: The model identifier for the agent being evaluated (e.g., 'azure/gpt-4.1')
        session: The session class to use for agent interactions (e.g., Tau2BenchSession)
        adapter: Adapter to convert agent trajectories to standardized trace format
        max_turns: Maximum number of conversation turns allowed per evaluation
        user_proxy_model: The model to use for simulating user responses
        user_proxy_persona: Type of user behavior to simulate ('expert' or 'non-expert')
        judge_model: The LLM-as-a-judge model used to score subgoal completion
        output_dir: Directory where evaluation results will be saved
        max_workers: Number of samples to evaluate concurrently
        n_trials: Number of times to run each sample for statistical reliability
    """

    agent_type: str
    session: Type[BaseSession]  # Class type, not instance
    adapter: BaseAdapter
    max_turns: int = 15
    user_proxy_model: str = "gpt-4.1"
    user_proxy_persona: str = "expert"
    judge_model: str = "gpt-5.4"
    output_dir: str = "experiment_outputs"
    max_workers: int = 1
    n_trials: int = 1
    k_value: int = n_trials

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


@dataclass
class EvaluationResult:
    """
    Container for the results of a single evaluation run.

    Captures all relevant information from evaluating one sample in one trial,
    including the conversation trajectory, computed metrics, and any errors encountered.

    Attributes:
        trial_id: The trial number for this evaluation (used when running multiple trials)
        sample_id: Unique identifier for the sample being evaluated
        run_id: Unique identifier for this specific evaluation run
        status: Execution status ('success' or 'failed')
        trajectory: Complete record of agent actions and observations during the run
        total_turns: Number of conversation turns that occurred
        error: Error message if the run failed, None otherwise
        metrics: Computed performance metrics (progress rates, etc.)
    """

    trial_id: int
    sample_id: str
    run_id: str
    status: str = "success"
    trajectory: Optional[Dict[str, Any]] = None
    total_turns: int = 0
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self, include_trajectory: bool = False) -> Dict[str, Any]:
        """Convert to dict. Set include_trajectory=True to include full trajectory data."""
        result = {}
        for k, v in self.__dict__.items():
            # Skip trajectory unless explicitly requested
            if k == "trajectory" and not include_trajectory:
                continue
            # Include non-None values and specific fields
            if v is not None or k in ["status", "total_turns", "trial_id"]:
                result[k] = v
        return result


async def run_sample(
    sample: Dict[str, Any], config: EvaluationConfig, trial_id: int = 0
) -> EvaluationResult:
    """
    Execute a single agent evaluation run for one sample.

    This function orchestrates the entire evaluation process: starting a session with the agent,
    creating a user proxy that simulates user behavior, running the conversation turn by turn,
    collecting the trajectory, and computing metrics.

    Args:
        sample: The evaluation sample containing task description and terminating conditions
        config: Configuration parameters for the evaluation
        trial_id: Trial number (for running the same sample multiple times)

    Returns:
        EvaluationResult containing trajectory, metrics, and execution status
    """
    sample_id = sample.get("id", str(uuid4()))
    run_id = str(uuid4())

    logger.info(
        f"[{sample_id}] Starting evaluation run {run_id}, (trial {trial_id}/{config.n_trials})"
    )

    # Extract sample metadata and convert to standardized format.
    # The tau2 airline dataset no longer carries a per-sample "domain" field;
    # the airline harness is always the "airline" domain.
    domain = sample.get("domain", "airline")

    # The dataset no longer provides a per-sample terminating condition; fall
    # back to a generic completion criterion when it is absent.
    DEFAULT_TERMINATING_CONDITION = (
        "The task is considered complete if the instruction goal is satisfied "
        "or you are transferred to another agent or you find yourself in a "
        "situation in which the scenario does not provide enough information "
        "for you to continue the conversation."
    )
    user_proxy_terminating_condition = sample.get(
        "terminating_condition", DEFAULT_TERMINATING_CONDITION
    )

    # Build the standardized evaluation sample from the current dataset schema
    # (top-level "subgoals" / "task_summary"), rather than the legacy
    # metadata.subgoals / input[0].content layout.
    sub_goals = [
        SubGoal(details=sg["description"]) for sg in sample.get("subgoals", [])
    ]
    sample = EvaluationSample(
        sub_goals=sub_goals,
        user_instruction=sample.get("task_summary", ""),
    )

    loop = asyncio.get_event_loop()

    try:
        # Create a new session instance for this evaluation run
        session = config.session(domain, config.agent_type)

        try:
            # Initialize the agent session
            await loop.run_in_executor(None, session.start_session)
            logger.debug(f"[{sample_id}] Session started: {session.session_id}")

            # Get description of agent capabilities for the user proxy
            agent_description = await loop.run_in_executor(None, session.get_agent_desc)

            # Create LLM client for the user proxy
            client = AzureOpenAIClient(
                model=config.user_proxy_model, max_tokens=1000, temperature=0
            )

            # Configure user proxy behavior based on persona
            if config.user_proxy_persona == "expert":
                user_proxy_is_expert = True
            elif config.user_proxy_persona == "non-expert":
                user_proxy_is_expert = False
            else:
                raise ValueError(f"Invalid user proxy persona: {config.user_proxy_persona}")

            logger.info(f"Using {config.user_proxy_persona} person for user proxy")

            # Create user proxy agent that will simulate user interactions
            user = UserProxyAgent(
                llm_client=client,
                task_summary=sample.user_instruction,
                terminating_conditions=[
                    TerminatingCondition(check=user_proxy_terminating_condition)
                ],
                agent_description=agent_description,
                config={USE_EXPERT_AGENT: user_proxy_is_expert},
            )

            # Initialize conversation tracking
            chat_history = ChatHistory(id=run_id, conversations=[])
            turn_count = 0

            # Main conversation loop: alternate between user proxy and agent
            while turn_count < config.max_turns:
                user_resp = await user.generate_message_from_chat_history(chat_history)
                user_message = user_resp.message_str

                if user_message is None:
                    logger.info(
                        f"[{sample_id}] User proxy did not generate a message. Ending conversation."
                    )
                    break

                if user_resp.check is not None:
                    logger.info(f"[{sample_id}] Terminating condition met")
                    break

                # Send user message to the agent and get response
                agent_resp_data = await loop.run_in_executor(
                    None, session.send_message, user_message
                )
                agent_resp = agent_resp_data.response_message

                # Record this turn in the conversation history
                chat_history.conversations.append(
                    ConversationTurn(
                        id=turn_count,
                        user_message=user_resp,
                        agent_responses=[ResponseFromAgent(response_str=agent_resp)],
                    )
                )

                logger.info(
                    f"[{sample_id}] Turn {turn_count} - User: {user_message} | Agent: {agent_resp}"
                )

                turn_count += 1

            # Retrieve the complete trajectory of agent actions
            trajectory = await loop.run_in_executor(None, session.get_trajectory)

            logger.info(f"Sample id: {sample_id} Trial ID {trial_id} Trajectory: {str(trajectory)}")

            logger.info(f"[{sample_id}] Completed. Total turns: {turn_count}")

            # Compute performance metrics from the trajectory
            metric_results = await loop.run_in_executor(
                None, calculate_metrics, trajectory, sample, config
            )

            return EvaluationResult(
                sample_id=sample_id,
                run_id=run_id,
                trial_id=trial_id,
                trajectory=trajectory,
                total_turns=turn_count,
                metrics=metric_results,
            )
        finally:
            # Always clean up the session, even if an error occurred
            try:
                await loop.run_in_executor(None, session.end)
            except Exception as e:
                logger.error(f"[{sample_id}] Error ending session: {e}")

    except Exception as e:
        logger.error(f"[{sample_id}] Evaluation run failed: {e}")
        return EvaluationResult(
            sample_id=sample_id,
            run_id=run_id,
            trial_id=trial_id,
            status="failed",
            error=str(e),
        )


async def run_evaluation_async(
    samples_file: str, config: EvaluationConfig
) -> List[EvaluationResult]:
    """
    Run evaluations on all samples with controlled concurrency.

    Loads samples from a JSON file and evaluates each one using the specified configuration.
    All (sample, trial) combinations are submitted to a single asyncio.gather pool, so
    trials across different samples can interleave freely. The semaphore caps total
    concurrent runs at max_workers regardless of trial boundaries.

    Args:
        samples_file: Path to JSON file containing evaluation samples
        config: Configuration parameters for the evaluation

    Returns:
        List of EvaluationResult objects, one for each sample-trial combination
    """
    # Load and validate samples file
    samples_path = Path(samples_file)
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")

    try:
        with open(samples_path, "r") as f:
            samples = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from samples file: {e}")

    if not isinstance(samples, list):
        samples = [samples]

    logger.info(f"Loaded {len(samples)} samples from {samples_file}")

    if not samples:
        logger.warning("No samples provided for evaluation.")
        return []

    total_tasks = len(samples) * config.n_trials
    logger.info(
        f"Running {len(samples)} samples x {config.n_trials} trials = {total_tasks} tasks "
        f"with max_workers={config.max_workers}"
    )

    # Single semaphore bounds total concurrency across all (sample, trial) pairs
    semaphore = asyncio.Semaphore(config.max_workers)

    async def run_with_semaphore(sample: Dict[str, Any], trial_id: int) -> EvaluationResult:
        """Wrapper to run sample evaluation with concurrency control."""
        async with semaphore:
            return await run_sample(sample, config, trial_id=trial_id)

    # Flatten: run all (sample, trial) combinations concurrently
    tasks = [
        run_with_semaphore(sample, trial_id=trial_id + 1)
        for trial_id in range(config.n_trials)
        for sample in samples
    ]

    all_results = await asyncio.gather(*tasks)

    successful = sum(1 for r in all_results if r.status == "success")
    failed = len(all_results) - successful
    logger.info(f"All tasks completed - Successful: {successful}, Failed: {failed}")

    return all_results


def run_evaluation(samples_file: str, config: EvaluationConfig) -> List[EvaluationResult]:
    """
    Run evaluation synchronously by creating a new event loop.

    This is a convenience wrapper around run_evaluation_async for synchronous code.
    Cannot be called from an async context - use run_evaluation_async directly instead.

    Args:
        samples_file: Path to JSON file containing evaluation samples
        config: Configuration parameters for the evaluation

    Returns:
        List of EvaluationResult objects

    Raises:
        Exception: If called from within an async context
    """
    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()
        raise Exception(
            "run_evaluation() cannot be called from within an async context. "
            "Use 'await run_evaluation_async(samples_file, config)' instead."
        )
    except RuntimeError:
        # If get_running_loop() raised RuntimeError, no loop is running - we can create one
        pass

    return asyncio.run(run_evaluation_async(samples_file, config))


def save_trial_results(results: List[EvaluationResult], config: EvaluationConfig) -> List[str]:
    """
    Save evaluation results grouped by trial to separate JSON files.

    Each output file contains all samples for one trial, including their trajectories,
    computed metrics, and metadata. This organization facilitates per-trial analysis.

    Args:
        results: List of all evaluation results from all trials
        config: Configuration used for the evaluation

    Returns:
        List of paths to saved trial result files
    """
    # Create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group results by trial_id
    results_by_trial = defaultdict(list)
    for r in results:
        results_by_trial[r.trial_id].append(r)

    saved_files = []

    # Save each trial's results to a separate file
    for trial_id, trial_results in sorted(results_by_trial.items()):
        trial_file = output_path / f"trial_{trial_id}_results.json"

        # Build trial data structure with metadata and all sample results
        trial_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_model": config.agent_type,
            "agent": config.session.__name__,
            "max_turns": config.max_turns,
            "user_proxy_model": config.user_proxy_model,
            "user_proxy_persona": config.user_proxy_persona,
            "judge_model": config.judge_model,
            "n_trials": config.n_trials,
            "trial_id": trial_id,
            "total_samples": len(trial_results),
            "successful": sum(1 for r in trial_results if r.status == "success"),
            "failed": sum(1 for r in trial_results if r.status == "failed"),
            "samples": [r.to_dict(include_trajectory=True) for r in trial_results],
        }

        trial_file.write_text(json.dumps(trial_data, indent=2, default=str))
        saved_files.append(str(trial_file))
        logger.info(f"Saved trial {trial_id} results to {trial_file}")

    return saved_files


def save_aggregate_metrics(
    aggregate_metrics: Dict[str, Any], summary: Dict[str, Any], config: EvaluationConfig
) -> str:
    """
    Save aggregate metrics and summary statistics to a JSON file.

    Combines aggregate metrics (Max@k across trials) with summary statistics
    and saves to a single file for easy reference.

    Args:
        aggregate_metrics: Calculated aggregate metrics across all trials
        summary: Overall summary statistics
        config: Configuration used for the evaluation

    Returns:
        Path to the saved aggregate metrics file
    """
    # Create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_model": config.agent_type,
        "agent": config.session.__name__,
        "max_turns": config.max_turns,
        "user_proxy_model": config.user_proxy_model,
        "user_proxy_persona": config.user_proxy_persona,
        "judge_model": config.judge_model,
        "n_trials": config.n_trials,
    }

    # Exclude metrics_by_sample from summary for JSON output
    summary_for_json = {k: v for k, v in summary.items() if k != "metrics_by_sample"}

    output = {**metadata, "summary": summary_for_json, **aggregate_metrics}

    metrics_file = output_path / "aggregate_metrics_results.json"
    metrics_file.write_text(json.dumps(output, indent=2, default=str))

    logger.info(f"Saved aggregate metrics to {metrics_file}")
    return str(metrics_file)


def extract_trajectory_stats(trajectory: Any) -> Dict[str, int]:
    """
    Extract token usage and tool call count from a trajectory.

    Args:
        trajectory: List of turns, where each turn is a list of messages

    Returns:
        Dictionary with total_tokens and tool_call_count
    """
    total_tokens = 0
    tool_call_count = 0

    if not isinstance(trajectory, list):
        return {"total_tokens": 0, "tool_call_count": 0}

    for turn in trajectory:
        if not isinstance(turn, list):
            continue
        for msg in turn:
            if not isinstance(msg, dict) or msg.get("role") not in (
                "agent",
                "assistant",
            ):
                continue
            # Sum token usage
            usage = msg.get("usage")
            if usage:
                total_tokens += usage.get("prompt_tokens", 0)
                total_tokens += usage.get("completion_tokens", 0)
            # Count tool calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tool_call_count += len(tool_calls)

    return {"total_tokens": total_tokens, "tool_call_count": tool_call_count}


def calculate_metrics(
    trajectory: Any, evaluation_sample: str, config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Calculate the final progress score from an agent's trajectory.
    Evaluates all subgoals against the full trajectory in a single pass.

    This simplified version only calculates progress rate (no AUC/PPT).

    Args:
        trajectory: Complete record of agent actions and observations
        evaluation_sample: The sample being evaluated
        config: Configuration for the evaluation

    Returns:
        Dictionary containing progress_rates and subgoal_validations
    """
    logger.info("Calculating metrics")

    # Convert trajectory to standardized agent trace format
    dialogue_trace = config.adapter.convert_to_agent_trace(trajectory)

    llm_client = AzureOpenAIClient(model=config.judge_model, max_tokens=1000, temperature=0)

    validator_config = {
        INCLUDE_VALIDATION_RESULTS: True,
        INCLUDE_JUDGE_EXPLANATION: True,
        INCLUDE_PROMPT_SENT_TO_LLMJ: True,
        OPTIMIZE_JUDGE_TRIALS: False,
        TEMPLATE_SUBGOAL: DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_TEMPLATE_ONE_SUBGOAL,
    }

    goal_completion_validator = SubgoalCompletionValidatorExperiment(
        llm_client=llm_client, config=validator_config
    )

    full_trajectory = dialogue_trace.turns
    user_task = evaluation_sample.user_instruction
    sub_goals = evaluation_sample.sub_goals

    validation_results = []
    for sub_goal in sub_goals:
        validation_result = asyncio.run(
            goal_completion_validator.validate_dynamic(
                turn_traces=full_trajectory,
                sub_goal=sub_goal,
                user_instruction=user_task,
            )
        )
        validation_results.append(validation_result)

    progress_score = ProgressScore.get_progress_score_from_validation_results(validation_results)

    # Extract token usage and tool call count from trajectory
    trajectory_stats = extract_trajectory_stats(trajectory)

    return {
        "progress_rates": [progress_score.score],
        "subgoal_validations": validation_results,
        "total_tokens": trajectory_stats["total_tokens"],
        "tool_call_count": trajectory_stats["tool_call_count"],
    }


def calculate_aggregate_metrics(
    results: List[EvaluationResult], config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Calculate Max@k aggregate metrics across multiple trials.

    For statistical robustness, each sample is run multiple times (trials).
    This function computes the maximum score achieved across all trials for each sample,
    then averages these maximums to get the overall Max@k metric.

    Args:
        results: List of all evaluation results from all trials
        config: Configuration used for the evaluation

    Returns:
        Dictionary containing Max@k metrics for progress score, including per-sample breakdowns
    """
    # Group results by sample_id to compare across trials
    results_by_sample = defaultdict(list)
    success_scores_by_sample = defaultdict(list)

    for r in results:
        if r.status == "success" and r.metrics:
            # Extract metric scores for this trial
            progress_rates = r.metrics.get("progress_rates") or []
            last_progress = progress_rates[-1] if progress_rates else None
            metrics_data = {
                "trial_id": r.trial_id,
                "progress_score": last_progress,
                "total_tokens": r.metrics.get("total_tokens"),
                "tool_call_count": r.metrics.get("tool_call_count"),
            }
            if last_progress is not None:
                progress_obj = NumericalScore(score=last_progress)
                success_obj = SuccessBasedMetric.get_success_score_from_progress_score(progress_obj)
                metrics_data["success_score"] = success_obj.score
                success_scores_by_sample[r.sample_id].append(success_obj)
            results_by_sample[r.sample_id].append(metrics_data)

    if not results_by_sample:
        logger.warning("No successful results with metrics found")
        return {}

    # Determine number of trials run
    n_trials = config.n_trials

    # Initialize aggregate metrics structure
    aggregate_metrics = {
        "n_samples": len(results_by_sample),
        "n_trials": n_trials,
        "metrics": {},
    }

    # Max Progress Rate: per trial -> progress_score -> max across trials -> avg across samples
    # Mean Progress Rate: per trial -> progress_score -> mean across trials -> avg across samples
    max_pr_per_sample = {}
    mean_pr_per_sample = {}
    for sample_id, sample_results in results_by_sample.items():
        trial_scores = [
            r["progress_score"] for r in sample_results if r["progress_score"] is not None
        ]
        max_pr_per_sample[sample_id] = max(trial_scores) if trial_scores else None
        mean_pr_per_sample[sample_id] = statistics.mean(trial_scores) if trial_scores else None

    valid_max_pr = [v for v in max_pr_per_sample.values() if v is not None]
    valid_mean_pr = [v for v in mean_pr_per_sample.values() if v is not None]

    aggregate_metrics["metrics"]["max_progress_rate"] = {
        "max_per_sample": max_pr_per_sample,
        "avg_max": statistics.mean(valid_max_pr) if valid_max_pr else None,
        "count": len(valid_max_pr),
    }
    aggregate_metrics["metrics"]["mean_progress_rate"] = {
        "per_sample": mean_pr_per_sample,
        "avg": statistics.mean(valid_mean_pr) if valid_mean_pr else None,
        "count": len(valid_mean_pr),
    }

    # Cost metrics: avg across trials (lower is better, so avg gives expected cost)
    for metric_name in ["total_tokens", "tool_call_count"]:
        avg_per_sample = {}
        for sid, res in results_by_sample.items():
            values = [r[metric_name] for r in res if r.get(metric_name) is not None]
            if values:
                avg_per_sample[sid] = statistics.mean(values)

        valid = list(avg_per_sample.values())
        aggregate_metrics["metrics"][metric_name] = {
            "avg_per_sample": avg_per_sample,
            "avg": statistics.mean(valid) if valid else None,
            "count": len(valid),
        }

    # Pass@K and PassHat@K calculation
    pass_at_k = PassAtK(
        config={
            K_VALUE: config.k_value,
            NO_OF_TRIALS: config.n_trials,
        }
    )
    pass_hat_k = PassHatK(
        config={
            K_VALUE: config.k_value,
            NO_OF_TRIALS: config.n_trials,
        }
    )
    k_val = config.k_value

    pass_at_k_per_sample = {}
    pass_hat_k_per_sample = {}

    for sample_id, success_objs in success_scores_by_sample.items():
        try:
            pass_at_k_value = pass_at_k.compute(success_objs)
            pass_at_k_per_sample[sample_id] = pass_at_k_value
        except Exception as e:
            logger.warning(f"[Warning] Skipping pass@{k_val} calculation for {sample_id}: {e}")
            pass_at_k_per_sample[sample_id] = None

        try:
            pass_hat_k_value = pass_hat_k.compute(success_objs)
            pass_hat_k_per_sample[sample_id] = pass_hat_k_value
        except Exception as e:
            logger.warning(f"[Warning] Skipping pass^{k_val} calculation for {sample_id}: {e}")
            pass_hat_k_per_sample[sample_id] = None

    valid_vals_pass_at_k = [v.score for v in pass_at_k_per_sample.values() if v is not None]
    pass_at_k_per_sample_scores = {
        k: (v.score if v is not None else None) for k, v in pass_at_k_per_sample.items()
    }

    aggregate_metrics["metrics"]["pass@k"] = {
        "k": k_val,
        "per_sample": pass_at_k_per_sample_scores,
        "avg": statistics.mean(valid_vals_pass_at_k) if valid_vals_pass_at_k else None,
        "count": len(valid_vals_pass_at_k),
    }

    valid_vals_pass_hat_k = [v.score for v in pass_hat_k_per_sample.values() if v is not None]
    pass_hat_k_per_sample_scores = {
        k: (v.score if v is not None else None) for k, v in pass_hat_k_per_sample.items()
    }

    aggregate_metrics["metrics"]["pass^k"] = {
        "k": k_val,
        "per_sample": pass_hat_k_per_sample_scores,
        "avg": statistics.mean(valid_vals_pass_hat_k) if valid_vals_pass_hat_k else None,
        "count": len(valid_vals_pass_hat_k),
    }

    return aggregate_metrics


def log_aggregate_metrics(aggregate_metrics: Dict[str, Any]) -> None:
    """
    Log aggregate metrics in a formatted, human-readable way.

    Displays Max@k metrics for each metric type, showing both overall averages
    and per-sample breakdowns for detailed analysis.

    Args:
        aggregate_metrics: Dictionary of calculated aggregate metrics
    """
    if not aggregate_metrics:
        return

    n_trials = aggregate_metrics["n_trials"]
    n_samples = aggregate_metrics["n_samples"]
    k = aggregate_metrics.get("k", n_trials)

    logger.info(f"{'=' * 70}")
    logger.info(f"Aggregate Metrics ({n_samples} samples)")
    logger.info(f"{'=' * 70}")

    # Map internal metric names to readable display names
    metric_display_names = {
        "pass@k": "Pass@K",
        "pass^k": "Pass^K",
        "mean_progress_rate": "MeanProgRate",
        "max_progress_rate": "ProgRate",
    }

    # Log summary statistics for each metric
    for metric_name, metric_data in aggregate_metrics["metrics"].items():
        display_name = metric_display_names.get(metric_name, metric_name)
        if "avg_max" in metric_data:
            avg_max = metric_data["avg_max"]
            count = metric_data["count"]

            if avg_max is not None:
                logger.info(f"Average Max{display_name}@{n_trials}:")
                logger.info(f"  Average: {avg_max:.4f} ({count}/{n_samples} samples)")
            else:
                logger.info(f"Average Max{display_name}@{n_trials}: No data available")

        elif "avg" in metric_data:
            avg = metric_data["avg"]
            count = metric_data["count"]
            if "k" in metric_data:
                logger.info(f"Average {display_name} for k:{k}:")
            else:
                logger.info(f"Average {display_name}@{n_trials}:")
            logger.info(
                f"  Average: {avg:.4f} ({count}/{n_samples} samples)"
                if avg is not None
                else "  No data available"
            )

    logger.info(f"{'-' * 70}")
    logger.info("Per-Sample Breakdown:")
    logger.info(f"{'-' * 70}")

    # Collect all sample IDs across all metrics
    all_sample_ids = set()
    for metric_data in aggregate_metrics["metrics"].values():
        if "max_per_sample" in metric_data:
            all_sample_ids.update(metric_data["max_per_sample"].keys())
        elif "per_sample" in metric_data:
            all_sample_ids.update(metric_data["per_sample"].keys())

    for sample_id in sorted(all_sample_ids):
        logger.info(f"Sample {sample_id}:")
        for metric_name, metric_data in aggregate_metrics["metrics"].items():
            display_name = metric_display_names.get(metric_name, metric_name)
            if "max_per_sample" in metric_data:
                val = metric_data["max_per_sample"].get(sample_id)
                label = f"Max{display_name}@{k}"
            elif "per_sample" in metric_data:
                val = metric_data["per_sample"].get(sample_id)
                label = f"{display_name} for k: {k}"
            else:
                logger.info(f"  Max{display_name}@{n_trials}: N/A")
                continue

            logger.info(
                f"  {label}: {val:.4f}" if isinstance(val, (int, float)) else f"  {label}: N/A"
            )


def create_summary(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Create summary statistics from all evaluation results.

    Compiles high-level statistics about the evaluation run, including success rates,
    average conversation length, failed runs, and per-sample metric breakdowns.

    Args:
        results: List of all evaluation results

    Returns:
        Dictionary containing summary statistics
    """
    # Calculate overall statistics
    n_trials = max((r.trial_id for r in results), default=1) if results else 1
    n_samples = len(set(r.sample_id for r in results))

    # Separate successful and failed runs
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    # Calculate average conversation length across successful runs
    avg_turns = sum(r.total_turns for r in successful) / len(successful) if successful else 0

    # Collect information about failed runs for debugging
    failed_runs = [
        {"sample_id": r.sample_id, "trial_id": r.trial_id, "error": r.error} for r in failed
    ]

    # Organize metrics by sample to show all trials for each sample
    metrics_by_sample = {}
    results_by_sample = defaultdict(list)
    for r in results:
        if r.status == "success":
            results_by_sample[r.sample_id].append(r)

    for sample_id, sample_results in results_by_sample.items():
        metrics_by_sample[sample_id] = [
            {
                "trial_id": r.trial_id,
                "progress_score": r.metrics.get("progress_rates", [None])[-1]
                if r.metrics
                else None,
            }
            for r in sample_results
            if r.metrics
        ]

    return {
        "n_trials": n_trials,
        "n_samples": n_samples,
        "total_runs": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "avg_turns": avg_turns,
        "failed_runs": failed_runs,
        "metrics_by_sample": metrics_by_sample,
    }


def log_summary(summary: Dict[str, Any]) -> None:
    """
    Log summary statistics in a readable format.

    Displays overall evaluation statistics, failed runs, and per-sample metrics
    to provide a complete picture of the evaluation results.

    Args:
        summary: Dictionary containing summary statistics
    """
    # Log overall statistics
    logger.info(f"Total samples:     {summary['n_samples']}")
    logger.info(f"Trials per sample: {summary['n_trials']}")
    logger.info(f"Total runs:        {summary['total_runs']}")
    logger.info(f"Successful:        {summary['successful']}")
    logger.info(f"Failed:            {summary['failed']}")

    if summary["successful"] > 0:
        logger.info(f"Average turns:     {summary['avg_turns']:.2f}")

    logger.info(f"{'-' * 70}")

    # Display details of any failed runs for debugging
    if summary["failed_runs"]:
        logger.info("Failed runs:")
        for failed_run in summary["failed_runs"]:
            logger.info(
                f"  - {failed_run['sample_id']} (trial {failed_run['trial_id']}): {failed_run['error']}"
            )

    # Display detailed metrics for each sample across all trials
    if summary["metrics_by_sample"]:
        logger.info("Metrics by sample:")
        for sample_id, trial_metrics in summary["metrics_by_sample"].items():
            logger.info(f"  Sample {sample_id} ({len(trial_metrics)} trial(s)):")
            for trial_metric in trial_metrics:
                progress = (
                    trial_metric["progress_score"]
                    if trial_metric["progress_score"] is not None
                    else "N/A"
                )
                logger.info(f"    Trial {trial_metric['trial_id']}: Progress={progress}")


def form_error_analysis_data_samples(
    results: List[EvaluationResult],
) -> List[SubgoalErrorAnalysisDataSample]:
    """
    Convert evaluation results to error analysis data format.

    Extracts subgoal validation results from evaluation outcomes and packages them
    in the format needed for error analysis.

    Args:
        results: List of evaluation results

    Returns:
        List of SubgoalErrorAnalysisDataSample objects ready for error analysis
    """
    return [
        SubgoalErrorAnalysisDataSample(
            data_sample_id=r.sample_id,
            agent_run_id=r.trial_id,
            subgoal_validations=r.metrics.get("subgoal_validations", []) if r.metrics else [],
        )
        for r in results
    ]


def analyse_errors(
    data_samples: List[SubgoalErrorAnalysisDataSample],
) -> SubgoalErrorAnalysisResult:
    """
    Perform automated error analysis on evaluation results.

    Uses LLM-based analysis to categorize and cluster errors encountered during
    agent evaluation, providing insights into common failure patterns.

    Args:
        data_samples: List of error analysis data samples

    Returns:
        SubgoalErrorAnalysisResult containing categorized and clustered errors
    """
    logger.info(f"Performing error analysis on {len(data_samples)} total samples across trials")
    client = AzureOpenAIClient(model="gpt-4.1", max_tokens=20000, temperature=1.0)

    error_analyser = UnsupervisedSubgoalErrorAnalysis(llm_client=client)
    analysed_results = error_analyser.analyze_batch(data_samples)

    return analysed_results


def parse_args():
    """
    Parse command-line arguments for configuring evaluation experiments.

    Defines all available command-line options for customizing the evaluation,
    including agent selection, user proxy configuration, parallelization settings,
    and output preferences.

    Returns:
        Parsed argument namespace with all configuration options
    """
    parser = argparse.ArgumentParser(
        description="Run agent evaluation experiments (simplified - progress rate only)"
    )

    parser.add_argument(
        "--agent-model",
        type=str,
        default="azure/gpt-4.1",
        help="Agent model to use (e.g., 'azure/gpt-4.1')",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="tau2bench",
        help="Agent type/session to use (options: 'tau2bench', 'toolsandbox') (default: 'tau2bench')",
    )

    parser.add_argument(
        "--terminating-msg",
        type=str,
        default="END_CONVERSATION",
        help="Terminating messages (default: 'END_CONVERSATION')",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum number of conversation turns (e.g,. 3) (default: 15 for tau2bench, 8 for toolsandbox agent)",
    )

    parser.add_argument(
        "--user-proxy-model",
        type=str,
        default="gpt-4.1",
        help="User proxy model to use (default: 'gpt-4.1')",
    )

    parser.add_argument(
        "--user-proxy-persona",
        type=str,
        default="expert",
        help="User proxy persona to use (options: 'expert', 'non-expert') (default: 'expert')",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.4",
        help="LLM-as-a-judge model used to score subgoal completion (default: 'gpt-5.4')",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_outputs",
        help="Output directory for results (default: 'experiment_outputs'), will be created in the paper_experiments folder with timestamp appended",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        help="Number of parallel samples to run (e.g,. 12) (always 1 for toolsandbox agent and default 3 for tau2bench agent if not specified)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        help="Number of trials per sample (default: 20 for tau2bench, 8 for toolsandbox agent)",
    )

    parser.add_argument(
        "--samples-file",
        type=str,
        default="paper_experiments/datasets/sample.json",
        help="Path to samples JSON file located at the project root (default: 'paper_experiments/datasets/sample.json')",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Main entry point for running agent evaluations.

    This script orchestrates the complete evaluation pipeline:
    1. Runs evaluations on all samples with specified configuration
    2. Saves per-trial results to JSON files
    3. Computes and saves summary statistics and aggregate metrics
    4. Performs automated error analysis and saves results
    """

    # Parse command-line arguments to configure the experiment
    args = parse_args()

    # Set up output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir.startswith("/"):
        # Absolute path - use as-is (for external callers like tau2bench evolution)
        output_dir = args.output_dir
    else:
        # Relative path - prepend paper_experiments/ and add timestamp (original behavior)
        output_dir = f"paper_experiments/{args.output_dir}_{timestamp}"

    # Configure evaluation based on agent type
    # Fields default to the values used in the paper if not explicitly set
    if args.agent.lower() == "tau2bench":
        config = EvaluationConfig(
            agent_type=args.agent_model,
            session=Tau2BenchSession,
            adapter=Tau2BenchAdapter(),
            max_turns=args.max_turns if args.max_turns else 15,
            user_proxy_model=args.user_proxy_model,
            user_proxy_persona=args.user_proxy_persona,
            judge_model=args.judge_model,
            output_dir=output_dir,
            max_workers=args.max_workers if args.max_workers else 3,
            n_trials=args.n_trials if args.n_trials else 20,
            k_value=args.n_trials if args.n_trials else 20,
        )
    elif args.agent.lower() == "toolsandbox":
        config = EvaluationConfig(
            agent_type=args.agent_model,
            session=ToolsandboxSession,
            adapter=ToolsandboxAdapter(),
            max_turns=args.max_turns if args.max_turns else 8,
            user_proxy_model=args.user_proxy_model,
            user_proxy_persona=args.user_proxy_persona,
            judge_model=args.judge_model,
            output_dir=output_dir,
            max_workers=args.max_workers
            if args.max_workers
            else 3,  # toolsandbox now supports parallel execution
            n_trials=args.n_trials if args.n_trials else 8,
            k_value=args.n_trials if args.n_trials else 8,
        )
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")

    # Set up logging to file and console
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(log_file))],
    )
    logger.info(f"Logging to {log_file}")

    try:
        # Step 1: Run evaluation for all samples using the specified configuration
        results = run_evaluation(args.samples_file, config)

        # Step 2: Save each trial's results to individual JSON files for per-trial analysis
        saved_files = save_trial_results(results, config)
        logger.info(f"Saved {len(saved_files)} trial result files")

        # Step 3: Create summary statistics and calculate Max@k aggregate metrics
        summary = create_summary(results)
        aggregate_metrics = calculate_aggregate_metrics(results, config)

        metrics_file = save_aggregate_metrics(aggregate_metrics, summary, config)

        # Display summary and aggregate metrics in log
        log_summary(summary)
        log_aggregate_metrics(aggregate_metrics)

        # Save raw results to pickle file for debugging and detailed analysis
        with open(f"{output_dir}/evaluation_results.pkl", "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved evaluation results to {output_dir}/evaluation_results.pkl")

        # Step 4: Perform automated error analysis to identify and cluster failure patterns
        error_analysis_data_samples = form_error_analysis_data_samples(results)
        error_analysis_results = analyse_errors(error_analysis_data_samples)

        # Save error analysis results for later visualization in UI
        with open(f"{output_dir}/error_analysis.pkl", "wb") as f:
            pickle.dump((error_analysis_data_samples, error_analysis_results), f)
        logger.info(f"Saved error analysis results to {output_dir}/error_analysis.pkl")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
