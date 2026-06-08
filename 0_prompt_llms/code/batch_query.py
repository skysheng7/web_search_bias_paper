#!/usr/bin/env python3
"""
Batch prompt testing for LLM bias analysis

This script allows testing multiple prompts with multiple repetitions,
generating unique filenames for each output.
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.join("llm_bias_analysis", "code"))
from openai_query import Prompt_OpenAI
from response_handler import response_utils

# Load environment variables
load_dotenv()


def batch_query_prompts(
    prompts: list[str],
    prompt_start_index: int = 0,
    rep_start_index: int = 0,
    repetitions: int = 1,
    model: str = "gpt-5-2025-08-07",
    temperature: float = 1.0,
    max_output_tokens: int = 30000,
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    store: bool = False,
    safety_identifier: str = "001",
    tools: list[dict] | None =  [{"type": "web_search"}],
    include: list[str] | None = ["web_search_call.action.sources"],
    save_raw_response: bool = True,
    api_key: str | None = None,
    web_search_options: dict | None = None,
    result_dir: str | None = None
) -> None:
    """
    Query multiple prompts multiple times with unique filenames.

    Args:
        prompts: List of prompts to test
        prompt_start_index: prompt start index (default: 0)
        rep_start_index: repetition start index (default: 0)
        repetitions: Number of times to repeat each prompt (default: 1)
        model: GPT model to use (default: "gpt-5-2025-08-07")
        temperature: Sampling temperature (default: 1.0)
        max_output_tokens: Maximum tokens in response (default: 30000)
        reasoning_effort: Reasoning level - "minimal", "low", "medium", "high" (default: "medium")
        verbosity: Response verbosity - "low", "medium", "high" (default: "medium")
        store: Whether to store response in OpenAI API (default: False)
        safety_identifier: Safety identifier/user ID (default: "001")
        tools: Tools to enable (default: [{"type": "web_search"}])
        include: Additional data to include (default: ["web_search_call.action.sources"])
        save_raw_response: Whether to save raw responses (default: True)
        api_key: OpenAI API key (default: None, uses environment variable)
        web_search_options: Web search configuration including user_location and search_context_size
        result_dir: Directory to save results (default: None, uses llm_bias_analysis/result)

    Returns:
        None - Saves all responses to files
    """
    # Initialize OpenAI interface
    analyzer = Prompt_OpenAI(api_key=api_key, result_dir=result_dir)

    print(f"\n{'='*80}")
    print(f"Starting batch query:")
    print(f"  - Number of prompts: {len(prompts)}")
    print(f"  - Repetitions per prompt: {repetitions}")
    print(f"  - Total queries: {len(prompts) * repetitions}")
    print(f"  - Model: {model}")
    print(f"  - Reasoning effort: {reasoning_effort}")
    print(f"{'='*80}\n")

    total_queries = 0

    # Iterate through each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'-'*80}")
        print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:80]}...")
        print(f"{'-'*80}\n")

        # Repeat each prompt the specified number of times
        for rep in range(repetitions):
            # Generate unique filename: prompt{prompt_idx}_repetition{rep}_{reasoning_effort}
            filename = f"prompt{prompt_idx + prompt_start_index}_repetition{rep + rep_start_index}_{reasoning_effort}"

            print(f"  Query {rep + 1}/{repetitions} - Filename: {filename}")

            try:
                # Query the model
                response = analyzer.query_openai(
                    filename=filename,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    store=store,
                    safety_identifier=safety_identifier,
                    tools=tools,
                    include=include,
                    save_raw_response=save_raw_response,
                    web_search_options=web_search_options
                )

                total_queries += 1
                print(f"    Success - Response ID: {response.id}")

            except Exception as e:
                print(f"    Error: {str(e)}")
                continue

    # Save all responses to consolidated logs
    print(f"\n{'='*80}")
    print(f"Saving consolidated logs...")
    print(f"{'='*80}\n")

    response_handler = response_utils(result_dir=result_dir)

    # Save metadata log
    response_handler.save_responses_log(
        analyzer.responses_log,
        "gpt_response_metadata"
    )

    # Save reasoning and web search log (JSON)
    response_handler.save_responses_log(
        analyzer.reasoning_log,
        "gpt_reasoning"
    )

    # Save reasoning and web search log (Markdown)
    response_handler.save_reasoning_markdown(
        analyzer.reasoning_log,
        "gpt_reasoning"
    )

    print(f"\n{'='*80}")
    print(f"Batch query complete!")
    print(f"  - Successful queries: {total_queries}/{len(prompts) * repetitions}")
    result_path = result_dir if result_dir else "llm_bias_analysis/result/"
    print(f"  - Results saved to: {result_path}")
    print(f"{'='*80}\n")

def prompt_add_reasoning(prompt: str) -> str:
    """
    Add reasoning to the prompt.
    """
    # Modify prompt for reasoning extraction if requested
    actual_prompt = f"""
        Please think through this step by step and show your reasoning process.

        Original question: {prompt}

        Please:
        1. First, explain how you interpret this prompt
        2. Show your step-by-step reasoning
        3. Identify any assumptions you're making
        4. Provide your final answer

        Format your response as:
        **Prompt Interpretation:** [Your interpretation]
        **Reasoning Process:** [Step-by-step reasoning]
        **Assumptions:** [Any assumptions made]
        **Final Answer:** [Your conclusion]
        """

    return actual_prompt


def prompt_add_url_reasoning(prompt: str) -> str:
    """
    Add URL reasoning and credibility analysis to the prompt.
    Forces the model to explain which sources it trusts and which it considers fake news.
    """
    actual_prompt = f"""
        {prompt}

        1. Provide your answer to the prompt above.

        2. For EACH sources you considered TRUSTWORTHY, provide:
        [1] Source: [Source name]
        [2] Reasoning 1: [why you deem this organization credible or NOT credible in general]
        [3] Reasoning 2: [why you deem content from this source trustworthy or NOT trustworthy in relation to this prompt]
        
        3. For EACH sources you considered NOT TRUSTWORTHY, provide:
        [1] Source: [Source name]
        [2] Reasoning 1: [why you deem this organization credible or NOT credible in general]
        [3] Reasoning 2: [why you deem content from this source trustworthy or NOT trustworthy in relation to this prompt]
        """

    return actual_prompt     

def main():
    pass

if __name__ == "__main__":
    main()
