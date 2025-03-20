import re
import logging
from typing import List, Dict, Any
from rl_loans.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


def extract_xml_answer(text: str) -> str:
    """
    Extract the answer between <recommendation> and </recommendation> using regex.
    Returns an empty string if no match is found.
    """
    match = re.search(r"<recommendation>(.*?)</recommendation>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def setup_debug_reward_functions():
    """Set up reward functions used for debugging/telemetry only (zero weights)"""
    
    # Telemetry reward functions (weight=0) for debugging
    def count_yes_responses(completions, **kwargs) -> List[float]:
        """Count how many YES responses are generated (telemetry only)"""
        counts = []
        for completion in completions:
            text = completion[0]["content"]
            answer = extract_xml_answer(text)
            counts.append(1.0 if answer.lower() == "approve" else 0.0)
        return counts
    
    def count_no_responses(completions, **kwargs) -> List[float]:
        """Count how many NO responses are generated (telemetry only)"""
        counts = []
        for completion in completions:
            text = completion[0]["content"]
            answer = extract_xml_answer(text)
            counts.append(1.0 if answer.lower() == "reject" else 0.0)
        return counts
    
    def count_yes_ground_truth(prompts, completions, answer, **kwargs) -> List[float]:
        """Count how many YES ground truth answers are seen (telemetry only)"""
        return [1.0 if a.lower() == "approve" else 0.0 for a in answer]
    
    def count_no_ground_truth(prompts, completions, answer, **kwargs) -> List[float]:
        """Count how many NO ground truth answers are seen (telemetry only)"""
        return [1.0 if a.lower() == "reject" else 0.0 for a in answer]

    debug_funcs = [
        count_yes_responses,
        count_no_responses,
        count_yes_ground_truth,
        count_no_ground_truth,
    ]
    
    # All debug rewards have zero weight
    debug_weights = [0.0] * len(debug_funcs)
    
    return debug_funcs, debug_weights


def setup_reward_functions(cfg: ExperimentConfig):
    """Set up the main reward functions for training."""
    
    def xml_formatting_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward function that gives a score based on proper XML formatting.
        It awards points if exactly one <reasoning> and one <recommendation> block exist,
        and subtracts a penalty if there's extra content after the closing </recommendation> tag.
        """
        rewards = []
        for completion in completions:
            text = completion[0]["content"]
            score = 0.0

            # Count reasoning blocks using regex.
            reasoning_blocks = re.findall(r"<reasoning>.*?</reasoning>", text, re.DOTALL)
            if len(reasoning_blocks) == 1:
                score += 0.5

            # Count answer blocks using regex.
            answer_blocks = re.findall(r"<recommendation>.*?</recommendation>", text, re.DOTALL)
            if len(answer_blocks) == 1:
                score += 0.5

                # Check for extra trailing content after the </recommendation> tag.
                closing_index = text.rfind("</recommendation>")
                if closing_index != -1:
                    trailing = text[closing_index + len("</recommendation>"):].strip()
                    score -= len(trailing) * 0.001  # small penalty per extra character

            rewards.append(score)
        return rewards

    def yes_no_reward_func(completions, **kwargs) -> List[float]:
        """
        Reward function that gives a fixed reward if the extracted answer is "yes" or "no"
        (case-insensitive).
        """
        rewards = []
        for completion in completions:
            text = completion[0]["content"]
            answer = extract_xml_answer(text)
            rewards.append(1.0 if answer.lower() in {"approve", "reject"} else 0.0)
        return rewards

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
        """
        Reward function that gives a higher reward (1.0) if the extracted answer exactly
        matches the provided correct label (ignoring case), otherwise 0.
        """
        # Calculate rewards
        extracted_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
        rewards = [
            1.0 if extracted_answer.lower() == correct_answer.lower() else 0.0 
            for extracted_answer, correct_answer in zip(extracted_answers, answer)
        ]

        # Log results
        log_correctness_results(prompts, completions, extracted_answers, answer, rewards)

        return rewards

    def log_correctness_results(prompts, completions, extracted_answers, correct_labels, rewards):
        """Log detailed results for correctness evaluation"""

        # Count how many answers were properly parsed as yes/no
        parsed_correctly = [answer.lower() in {'approve', 'reject'} for answer in extracted_answers]
        parsed_count = sum(parsed_correctly)
        
        # Of the properly parsed ones, how many were correct
        correct_parsed = sum(
            1 for parsed, answer, correct in zip(parsed_correctly, extracted_answers, correct_labels)
            if parsed and answer.lower() == correct.lower()
        )
        parsed_accuracy = correct_parsed / parsed_count if parsed_count > 0 else 0.0

        # Log detailed results for debugging
        LOGGER.info("\nCorrectness Results:")
        for i, (prompt, completion, answer, correct, reward) in enumerate(
            zip(prompts, completions, extracted_answers, correct_labels, rewards)
        ):
            LOGGER.info(
                f"\nExample {i+1}:"
                f"\nPrompt: {repr(prompt)[:100]}..."
                f"\nCompletion: {repr(completion[0]['content'])}" # verbose logging
                f"\nExtracted: '{repr(answer)}' | Target: '{repr(correct)}' | Correct: {reward > 0}"
            )
        LOGGER.info(f"\nParsed correctly: {parsed_count}/{len(rewards)}")
        LOGGER.info(f"\nParsed accuracy: {parsed_accuracy*100:.1f}%")

    reward_funcs = [xml_formatting_reward_func, yes_no_reward_func, correctness_reward_func]
    reward_weights = [cfg.xml_formatting_reward_weight, cfg.yes_no_reward_weight, cfg.correctness_reward_weight]

    return reward_funcs, reward_weights