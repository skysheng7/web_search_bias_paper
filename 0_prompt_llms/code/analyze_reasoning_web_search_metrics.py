"""
Analyze reasoning and web search metrics from trial_1_reason_url data.

This module provides functions to analyze reasoning and web search metrics from
trial_1_reason_url data across US, UK, and Europe regions. It extracts:
- Empty reasoning responses count
- Empty web search responses count
- Total reasoning and web search counts per prompt
- URL statistics per web search query
- Token usage statistics per prompt

Results are saved as CSV files in each region's folder.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def create_full_template(region: str) -> pd.DataFrame:
    """
    Create complete template dataframe based on experimental design.

    The experimental design uses:
    - 10 politicians per region
    - 2 prompt versions per politician: 'disregards' (even prompt numbers) and 'values' (odd prompt numbers)
    - 5 repetitions per prompt version (0-4)
    - Prompt numbering: politician_index * 2 for 'disregards', politician_index * 2 + 1 for 'values'
    - Total: 10 politicians × 2 versions × 5 repetitions = 100 prompts per region

    Args:
        region (str): Region name ('US', 'UK', or 'Europe')

    Returns:
        pd.DataFrame: Complete template with all expected prompt combinations
    """
    # Import politicians data
    from politicians_data import POLITICIANS

    if region not in POLITICIANS:
        return pd.DataFrame()

    politicians = POLITICIANS[region]
    repetitions = list(range(5))  # 0, 1, 2, 3, 4

    # Generate all combinations
    template_data = []

    for politician_idx, politician in enumerate(politicians):
        # Each politician gets 2 prompt numbers: one for disregards, one for values
        disregards_prompt_num = politician_idx * 2
        values_prompt_num = politician_idx * 2 + 1

        # Generate prompts for 'disregards' version
        for repetition in repetitions:
            prompt_id = f"prompt{disregards_prompt_num}_repetition{repetition}_medium"
            template_data.append(
                {
                    "prompt_id": prompt_id,
                    "politician": politician,
                    "prompt_version": "disregards",
                    "region": region,
                }
            )

        # Generate prompts for 'values' version
        for repetition in repetitions:
            prompt_id = f"prompt{values_prompt_num}_repetition{repetition}_medium"
            template_data.append(
                {
                    "prompt_id": prompt_id,
                    "politician": politician,
                    "prompt_version": "values",
                    "region": region,
                }
            )

    template_df = pd.DataFrame(template_data)
    return template_df.sort_values("prompt_id").reset_index(drop=True)


def load_prompt_reasoning(region_path: Path, prompt_id: str) -> Dict[str, Any]:
    """
    Load reasoning and web search data from individual prompt JSON file.

    Args:
        region_path (Path): Path to the region directory
        prompt_id (str): Prompt identifier (e.g., 'prompt0_repetition0_medium')

    Returns:
        Dict[str, Any]: Prompt data containing detailed_response, or empty dict if not found
    """
    prompt_file = region_path / f"{prompt_id}.json"
    if not prompt_file.exists():
        return {}

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data
    except (json.JSONDecodeError, KeyError):
        return {}


def analyze_prompt_reasoning_web_search(prompt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze reasoning and web search data from a single prompt's data.

    Args:
        prompt_data (Dict[str, Any]): Prompt data from individual JSON file

    Returns:
        Dict[str, Any]: Dictionary containing:
            - empty_reasoning: Count of empty reasoning responses
            - empty_web_search: Count of empty web search responses
            - empty_reasoning_and_web_search: Sum of empty reasoning and empty web search
            - total_reasoning: Total number of reasoning responses
            - total_web_search: Total number of web search responses
            - median_urls_per_query: Median number of URLs per web search query
            - total_urls: Total number of URLs across all web searches
    """
    if not prompt_data:
        return {
            "empty_reasoning": 0,
            "empty_web_search": 0,
            "empty_reasoning_and_web_search": 0,
            "total_reasoning": 0,
            "total_web_search": 0,
            "median_urls_per_query": 0,
            "total_urls": 0,
        }

    # Initialize counters
    metrics = {
        "empty_reasoning": 0,
        "empty_web_search": 0,
        "empty_reasoning_and_web_search": 0,
        "total_reasoning": 0,
        "total_web_search": 0,
        "urls_per_query": [],
        "total_urls": 0,
    }

    # Get output data from the prompt response
    output_data = prompt_data.get("output", [])
    if not output_data:
        return _finalize_metrics(metrics)

    # Process each output entry
    for output_entry in output_data:
        _process_output_entry(output_entry, metrics)

    return _finalize_metrics(metrics)


def _finalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate derived metrics and clean up intermediate data.

    Args:
        metrics (Dict[str, Any]): Metrics dictionary with intermediate data

    Returns:
        Dict[str, Any]: Final metrics dictionary
    """
    # Calculate derived metrics
    metrics["empty_reasoning_and_web_search"] = (
        metrics["empty_reasoning"] + metrics["empty_web_search"]
    )
    metrics["median_urls_per_query"] = (
        np.median(metrics["urls_per_query"]) if metrics["urls_per_query"] else 0
    )

    # Remove intermediate data and return final metrics
    del metrics["urls_per_query"]
    return metrics


def _process_output_entry(
    output_entry: Dict[str, Any], metrics: Dict[str, Any]
) -> None:
    """
    Process a single output entry from prompt response.

    Args:
        output_entry (Dict[str, Any]): Single output entry from prompt JSON
        metrics (Dict[str, Any]): Metrics dictionary to update
    """
    entry_type = output_entry.get("type")

    if entry_type == "reasoning":
        metrics["total_reasoning"] += 1
        summary = output_entry.get("summary", [])
        if _is_empty_reasoning(summary):
            metrics["empty_reasoning"] += 1

    elif entry_type == "web_search_call":
        metrics["total_web_search"] += 1
        action = output_entry.get("action", {})

        if _is_empty_web_search_action(action):
            metrics["empty_web_search"] += 1

        # Count URLs from sources
        sources = action.get("sources", [])
        url_count = len(sources)
        metrics["urls_per_query"].append(url_count)
        metrics["total_urls"] += url_count


def _is_empty_web_search_action(action: Dict[str, Any]) -> bool:
    """
    Check if web search action is empty.

    Args:
        action (Dict[str, Any]): Web search action data

    Returns:
        bool: True if web search action is empty, False otherwise
    """
    return (
        not action
        or action.get("query") in ["N/A", "", None]
        or not action.get("sources", [])
    )


def _is_empty_reasoning(reasoning_content: Any) -> bool:
    """
    Check if reasoning content is empty.

    Args:
        reasoning_content: Reasoning content to check

    Returns:
        bool: True if reasoning is empty, False otherwise
    """
    return not reasoning_content or (
        isinstance(reasoning_content, list) and len(reasoning_content) == 0
    )


def extract_token_usage(prompt_data: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract token usage statistics from prompt data.

    Args:
        prompt_data (Dict[str, Any]): Prompt data from JSON file

    Returns:
        Dict[str, int]: Dictionary containing input_tokens, output_tokens, and total_tokens
    """
    if not prompt_data:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    usage = prompt_data.get("usage", {})
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def check_first_response_is_reasoning(prompt_data: Dict[str, Any]) -> bool:
    """
    Check if the first response in the output array is of type 'reasoning'.

    Args:
        prompt_data (Dict[str, Any]): Prompt data from JSON file

    Returns:
        bool: True if first response is reasoning, False otherwise
    """
    if not prompt_data:
        return False

    output_data = prompt_data.get("output", [])
    if not output_data or len(output_data) == 0:
        return False

    # Check the type of the first output entry
    first_entry = output_data[0]
    return first_entry.get("type") == "reasoning"


def check_first_response_is_empty_reasoning(prompt_data: Dict[str, Any]) -> bool:
    """
    Check if the first response in the output array is an empty reasoning.

    Args:
        prompt_data (Dict[str, Any]): Prompt data from JSON file

    Returns:
        bool: True if first response is empty reasoning, False otherwise
    """
    if not prompt_data:
        return False

    output_data = prompt_data.get("output", [])
    if not output_data or len(output_data) == 0:
        return False

    # Check the type of the first output entry
    first_entry = output_data[0]
    if first_entry.get("type") != "reasoning":
        return False

    # Check if the reasoning content is empty
    summary = first_entry.get("summary", [])
    return _is_empty_reasoning(summary)


def check_first_empty_reasoning_second_web_search(prompt_data: Dict[str, Any]) -> bool:
    """
    Check if the first response is empty reasoning and second response is a non-empty web search.

    Args:
        prompt_data (Dict[str, Any]): Prompt data from JSON file

    Returns:
        bool: True if first response is empty reasoning and second is non-empty web search, False otherwise
    """
    if not prompt_data:
        return False

    output_data = prompt_data.get("output", [])
    if not output_data or len(output_data) < 2:
        return False

    # Check if first response is empty reasoning
    first_entry = output_data[0]
    if first_entry.get("type") != "reasoning":
        return False

    summary = first_entry.get("summary", [])
    if not _is_empty_reasoning(summary):
        return False

    # Check if second response is web search with non-empty query
    second_entry = output_data[1]
    if second_entry.get("type") != "web_search_call":
        return False

    action = second_entry.get("action", {})
    # Check that the web search is NOT empty (has a valid query)
    return not _is_empty_web_search_action(action)


def process_region(region_path: Path) -> pd.DataFrame:
    """
    Process a single region and return results dataframe with all metrics.

    Args:
        region_path (Path): Path to the region directory (e.g., US, UK, Europe)

    Returns:
        pd.DataFrame: DataFrame containing all metrics for each prompt in the region
    """
    # Get region name from path
    region_name = region_path.name

    # Create complete template based on experimental design
    template_df = create_full_template(region_name)
    if template_df.empty:
        return pd.DataFrame()

    # Initialize results list
    results = []

    for _, row in template_df.iterrows():
        prompt_id = row["prompt_id"]

        # Check if JSON file exists before processing
        json_file = region_path / f"{prompt_id}.json"
        if not json_file.exists():
            # Skip this entry if the JSON file doesn't exist
            continue

        # Load individual prompt data (more memory efficient)
        prompt_data = load_prompt_reasoning(region_path, prompt_id)

        # Skip if prompt data is empty (failed to load)
        if not prompt_data:
            continue

        # Analyze reasoning and web search from prompt data
        reasoning_metrics = analyze_prompt_reasoning_web_search(prompt_data)

        # Extract token usage from same prompt data
        token_metrics = extract_token_usage(prompt_data)

        # Check if first response is reasoning
        first_response_is_reasoning = check_first_response_is_reasoning(prompt_data)

        # Check if first response is empty reasoning
        first_response_is_empty_reasoning = check_first_response_is_empty_reasoning(
            prompt_data
        )

        # Check if first response is empty reasoning and second is non-empty web search
        first_empty_reasoning_second_web_search = (
            check_first_empty_reasoning_second_web_search(prompt_data)
        )

        # Combine all metrics
        result = {
            "prompt_id": prompt_id,
            "politician": row["politician"],
            "prompt_version": row["prompt_version"],
            "region": row["region"],
            **reasoning_metrics,
            **token_metrics,
            "first_response_is_reasoning": first_response_is_reasoning,
            "first_response_is_empty_reasoning": first_response_is_empty_reasoning,
            "first_empty_reasoning_second_web_search": first_empty_reasoning_second_web_search,
        }

        results.append(result)

    return pd.DataFrame(results)


def save_region_metrics(results_df: pd.DataFrame, region_path: Path) -> str:
    """
    Save region metrics to CSV file.

    Args:
        results_df (pd.DataFrame): DataFrame containing metrics for the region
        region_path (Path): Path to the region directory

    Returns:
        str: Path to the saved CSV file
    """
    output_file = region_path / "reasoning_web_search_metrics.csv"
    results_df.to_csv(output_file, index=False)
    return str(output_file)


def calculate_region_summary(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics for a region.

    Args:
        results_df (pd.DataFrame): DataFrame containing metrics for the region

    Returns:
        Dict[str, float]: Dictionary containing summary statistics
    """
    return {
        "total_prompts": len(results_df),
        "avg_empty_reasoning": results_df["empty_reasoning"].mean(),
        "avg_empty_web_search": results_df["empty_web_search"].mean(),
        "avg_empty_reasoning_and_web_search": results_df[
            "empty_reasoning_and_web_search"
        ].mean(),
        "avg_total_reasoning": results_df["total_reasoning"].mean(),
        "avg_total_web_search": results_df["total_web_search"].mean(),
        "avg_median_urls_per_query": results_df["median_urls_per_query"].mean(),
        "avg_total_tokens": results_df["total_tokens"].mean(),
        "first_response_reasoning_count": results_df[
            "first_response_is_reasoning"
        ].sum(),
        "first_response_reasoning_percentage": results_df[
            "first_response_is_reasoning"
        ].mean()
        * 100,
        "first_response_empty_reasoning_count": results_df[
            "first_response_is_empty_reasoning"
        ].sum(),
        "first_response_empty_reasoning_percentage": results_df[
            "first_response_is_empty_reasoning"
        ].mean()
        * 100,
        "first_empty_reasoning_second_web_search_count": results_df[
            "first_empty_reasoning_second_web_search"
        ].sum(),
        "first_empty_reasoning_second_web_search_percentage": results_df[
            "first_empty_reasoning_second_web_search"
        ].mean()
        * 100,
    }


def process_all_regions(base_path: str = None) -> Dict[str, Dict[str, float]]:
    """
    Process all regions and return summary statistics.

    Args:
        base_path (str, optional): Base path to trial_1_reason_url directory.
                                  If None, uses default path.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping region names to their summary statistics
    """
    if base_path is None:
        base_path = "/Users/skysheng/Desktop/github/prompt_revision/0_prompt_llms/result/trial_1_reason_url"

    base_path = Path(base_path)
    regions = ["US", "UK", "Europe"]
    summaries = {}

    for region in regions:
        region_path = base_path / region

        if not region_path.exists():
            print(f"Warning: Region path does not exist: {region_path}")
            continue

        # Process region
        results_df = process_region(region_path)

        if not results_df.empty:
            # Save results
            output_file = save_region_metrics(results_df, region_path)
            print(f"Saved {region} results to {output_file}")

            # Calculate and store summary
            summary = calculate_region_summary(results_df)
            summaries[region] = summary

            # Print summary statistics
            print(f"\n=== Summary for {region} ===")
            print(f"Total prompts analyzed: {summary['total_prompts']}")
            print(
                f"Average empty reasoning per prompt: {summary['avg_empty_reasoning']:.2f}"
            )
            print(
                f"Average empty web search per prompt: {summary['avg_empty_web_search']:.2f}"
            )
            print(
                f"Average empty reasoning + web search per prompt: {summary['avg_empty_reasoning_and_web_search']:.2f}"
            )
            print(
                f"Average total reasoning per prompt: {summary['avg_total_reasoning']:.2f}"
            )
            print(
                f"Average total web search per prompt: {summary['avg_total_web_search']:.2f}"
            )
            print(
                f"Average median URLs per query: {summary['avg_median_urls_per_query']:.2f}"
            )
            print(f"Average total tokens per prompt: {summary['avg_total_tokens']:.0f}")
            print(
                f"First response is reasoning: {summary['first_response_reasoning_count']}/{summary['total_prompts']} "
                f"({summary['first_response_reasoning_percentage']:.1f}%)"
            )
            print(
                f"First response is empty reasoning: {summary['first_response_empty_reasoning_count']}/{summary['total_prompts']} "
                f"({summary['first_response_empty_reasoning_percentage']:.1f}%)"
            )
            print(
                f"First empty reasoning, second web search: {summary['first_empty_reasoning_second_web_search_count']}/{summary['total_prompts']} "
                f"({summary['first_empty_reasoning_second_web_search_percentage']:.1f}%)"
            )
        else:
            print(f"Warning: No results generated for {region}")

    return summaries


if __name__ == "__main__":
    # Example usage: process all regions and display summaries
    summaries = process_all_regions()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    for region, summary in summaries.items():
        print(f"\n{region}: {summary['total_prompts']} prompts processed")
