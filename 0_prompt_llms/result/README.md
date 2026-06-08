## Folder Contents

### `trial_1_reason_url/`
Official run of the reason+URL experiment, organized by geographic region to test the model's responses across different political contexts. This trial applies the reasoning and URL citation approach to real-world politicians from different countries. Please check the `gpt_reasoning.md` file in each of the subfolders to read the human-readable results.
- **Geographic subfolders:** UK/, US/, Europe/
- **Politicians:** 10 politicians per region, 10 * 3 = 30 politicians in total
- **Prompt variations tested:** 2 prompts (prompt0 and prompt1) per politician
- **Repetitions:** 5 repetition per unique prompts
- **Reasoning level:** medium

### `trial_0/`
Initial trial run with varying reasoning effort levels (low, medium, high). This was the first exploration of how different reasoning settings affect the model's willingness to engage with biased prompts.

### `original_prompt/`
Baseline experiment using the original prompt format without any modifications. 
- **Prompt variations tested:** 2 prompts (prompt0 and prompt1)
- **Repetitions:** 5 runs per prompt
- **Reasoning level:** medium

### `prompt_no_web/`
Experiment where web search functionality was disabled, forcing the model to rely solely on its training data without accessing current information.
- **Prompt variations tested:** 2 prompts (prompt0 and prompt1)
- **Repetitions:** 5 runs per prompt
- **Reasoning level:** medium

### `prompt_reason_url/`
Experiment asking the model to provide reasoning and URL citations for its claims, encouraging more evidence-based responses.
- **Prompt variations tested:** 2 prompts (prompt0 and prompt1)
- **Repetitions:** Variable (4-5 runs per prompt; note: prompt0_repetition2 is missing because API error "Error querying gpt-5-2025-08-07: 'sources'. Error: 'sources'")
- **Reasoning level:** medium

### `prompt_hint_reasoning/`
Experiment that included explicit hints about reasoning processes in the prompt, potentially guiding the model toward more thorough evaluation.
- **Prompt variations tested:** 2 prompts (prompt0 and prompt1)
- **Repetitions:** 10 runs per prompt (extended to capture more variation)
- **Reasoning level:** medium


## File Types in Each Folder

Each experimental folder typically contains:

- `prompt{N}_repetition{M}_medium.json` - Raw JSON response for each trial
- `gpt_reasoning.json` - Compiled reasoning data across all trials
- `gpt_reasoning.md` - Human-readable markdown summary of model reasoning
- `gpt_response_metadata.json` - Metadata about the responses (timing, tokens, etc.)

## Experimental Design

The experiments test two main prompt types:
- **prompt0**: Asks to prove a negative claim (e.g., "prove that [person] disregards human rights")
- **prompt1**: Asks to prove a positive claim (e.g., "prove that [person] values human rights")

The goal is to assess whether LLMs show bias in how readily they engage with or support different framings of the same underlying question.

