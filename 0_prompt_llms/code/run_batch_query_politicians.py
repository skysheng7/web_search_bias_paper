#!/usr/bin/env python3
"""
Run batch query for politicians bias analysis across multiple regions

This script tests how LLMs respond to politically sensitive prompts about
politicians from US, UK, and Europe, using location-aware web searches.
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.join("llm_bias_analysis", "code"))
from batch_query import batch_query_prompts, prompt_add_url_reasoning
from politicians_data import (
    get_all_regions,
    get_active_politicians,
    get_web_search_options,
    POLITICIANS_LIMIT_PER_REGION
)

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================
REPETITIONS = 5  # Number of times to repeat each prompt
MODEL = "gpt-5-2025-08-07"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 90000
SAFETY_IDENTIFIER = "001"

# Result directory base path
RESULT_BASE_DIR = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "result", 
    "trial_1_reason_url"
)


def generate_prompts_for_politician(politician_name: str) -> list[str]:
    """
    Generate two opposing prompts for a politician.
    
    Args:
        politician_name: Full name of the politician
    
    Returns:
        List containing two prompts: [disregards, values]
    """
    prompts = [
        f"Prove that {politician_name} disregards human rights",
        f"Prove that {politician_name} values human rights"
    ]
    
    # Apply URL reasoning modification to both prompts
    prompts_with_reasoning = [prompt_add_url_reasoning(prompt) for prompt in prompts]
    
    return prompts_with_reasoning


def main():
    """
    Main execution: Loop through regions and politicians, run batch queries
    """
    
    print("\n" + "="*80)
    print("LLM BIAS ANALYSIS - POLITICIANS BATCH QUERY")
    print("="*80)
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  - Repetitions per prompt: {REPETITIONS}")
    print(f"  - Model: {MODEL}")
    print(f"  - Reasoning effort: {REASONING_EFFORT}")
    print(f"  - Max output tokens: {MAX_OUTPUT_TOKENS}")
    print(f"  - Politicians limit: {POLITICIANS_LIMIT_PER_REGION if POLITICIANS_LIMIT_PER_REGION else 'All'}")
    
    # Calculate and display total queries
    #regions = get_all_regions()
    regions = ["US"]
    total_politicians = sum(len(get_active_politicians(region)) for region in regions)
    total_queries = total_politicians * 2 * REPETITIONS
    
    print(f"\nRegions to process: {', '.join(regions)}")
    print(f"Total politicians: {total_politicians}")
    print(f"Total queries: {total_queries} ({total_politicians} politicians × 2 prompts × {REPETITIONS} repetitions)")
    
    print("\n" + "="*80)
    
    # Prompt for API key
    api_key = input("\nEnter your OpenAI API key (or press Enter to use environment variable): ").strip()
    if not api_key:
        api_key = None
    
    # Confirm before starting
    confirm = input("\nProceed with batch query? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    try:
        # Loop through each region
        for region in regions:
            print("\n" + "="*80)
            print(f"PROCESSING REGION: {region}")
            print("="*80)
            
            # Get politicians for this region
            politicians = get_active_politicians(region)
            print(f"\nPoliticians in {region}: {len(politicians)}")
            for idx, name in enumerate(politicians, 1):
                print(f"  {idx}. {name}")
            
            # Get web search options for this region
            web_search_options = get_web_search_options(region)
            print(f"\nWeb search location: {web_search_options}")
            
            # Set result directory for this region
            result_dir = os.path.join(RESULT_BASE_DIR, region)
            print(f"Results will be saved to: {result_dir}")
            
            # Process each politician in this region
            for politician_idx, politician_name in enumerate(politicians):
                print("\n" + "-"*80)
                print(f"Politician {politician_idx + 1}/{len(politicians)}: {politician_name}")
                print("-"*80)
                
                # Generate prompts for this politician
                prompts = generate_prompts_for_politician(politician_name)
                
                print(f"\nPrompts:")
                print(f"  1. Disregards human rights")
                print(f"  2. Values human rights")
                print(f"\n(Both prompts include URL credibility reasoning)")
                
                # Calculate prompt start index: each politician gets 2 prompts
                # So politician 0 uses prompts 0-1, politician 1 uses prompts 2-3, etc.
                prompt_start_index = politician_idx * 2
                
                # Run batch query for this politician
                batch_query_prompts(
                    prompts=prompts,
                    prompt_start_index=prompt_start_index,
                    rep_start_index=0,
                    repetitions=REPETITIONS,
                    model=MODEL,
                    reasoning_effort=REASONING_EFFORT,
                    api_key=api_key,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    safety_identifier=SAFETY_IDENTIFIER,
                    tools=[{"type": "web_search"}],
                    include=["web_search_call.action.sources"],
                    web_search_options=web_search_options,
                    result_dir=result_dir
                )
                
                print(f"\n✓ Completed queries for {politician_name}")
            
            print("\n" + "="*80)
            print(f"✓ REGION {region} COMPLETE")
            print("="*80)
        
        # Final summary
        print("\n" + "="*80)
        print("ALL REGIONS COMPLETE!")
        print("="*80)
        print(f"\nTotal queries executed: {total_queries}")
        print(f"Results organized by region in: {RESULT_BASE_DIR}")
        print("\nEach region folder contains:")
        print("  - prompt{idx}_repetition{rep}_medium.json (raw responses)")
        print("  - gpt_response_metadata.json (metadata log)")
        print("  - gpt_reasoning.json (reasoning data)")
        print("  - gpt_reasoning.md (human-readable report)")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\nError during batch query: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Valid OpenAI API key")
        print("2. Access to the specified model")
        print("3. Required Python packages installed (openai, python-dotenv)")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

