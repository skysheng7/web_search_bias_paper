# LLM Bias Analysis - Code Documentation

## Helper Modules
- `openai_query.py` - Core functions to query OpenAI API and extract reasoning/web search data
- `response_handler.py` - Functions to save responses as JSON and Markdown files
- `batch_query.py` - Functions to run multiple prompts with repetitions and modify prompts
- `reconstruct_logs.py` - Functions to rebuild logs from raw OpenAI query response JSON files
- `politicians_data.py` - Dictionary of politicians by region and web search location settings

## Scripts to Run
- `run_single_query.py` - Test a single query (good for debugging)
- `run_batch_query_original_prompt.py` - Run original prompts about Keir Starmer with web search
- `run_batch_query_no_web.py` - Run prompts without web search access
- `run_batch_query_hint_reasoning.py` - Run prompts with explicit reasoning instructions added
- `run_batch_query_reason_url.py` - Run prompts that analyze URL credibility
- `run_batch_query_politicians.py` - Run large-scale bias test across 15 politicians from US/UK/Europe
- `run_reconstruct_logs.py` - Rebuild consolidated logs from raw files (useful if something breaks)

## Quick Start
1. Set `OPENAI_API_KEY` in `.env` file
2. Run `run_single_query.py` to test your setup
3. Run any of the batch scripts to conduct experiments
4. Check the `result/` folder for output JSON and Markdown files

