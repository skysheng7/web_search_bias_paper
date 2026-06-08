#!/usr/bin/env python3
"""
Prompt OpenAI reasoning models about political sensitive content for bias investigation

This script prompt OpenAI's LLMs, extract reasoning, and web search information.
"""

import os
import sys
import time
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
sys.path.append(os.path.join("llm_bias_analysis", "code"))
from response_handler import response_utils, reasoning_and_web_search_response

# Load environment variables from .env file
load_dotenv()

# dataclass to store metadata about each prompt & response
@dataclass
class GPTResponse_metadata:
    """Structure to hold GPT response data for analysis"""
    filename: str
    prompt: str
    response: str
    reasoning_num: int
    web_search_request_num: int
    web_source_num: int
    model: str
    timestamp: str
    temperature: float
    max_output_tokens: int
    reasoning_effort: str # None, "minimal", "low", "medium", "high"
    verbosity: str # "low", "medium", "high"
    store: bool
    safety_identifier: str | None
    tools: list[dict] | None
    include: list[str] | None
    prompt_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    id: str
    web_search_options: dict | None

class Prompt_OpenAI:
    """
    Interface for querying OpenAI API and analyzing model bias.

    Handles GPT model queries with various settings, extracts reasoning and web search
    data, and logs responses for analysis.
    """
    
    def __init__(self, api_key: str | None = None, result_dir: str | None = None):
        """
        Initialize the GPT interface
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment
            result_dir: Directory to save results. If None, uses default location
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Please set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Set result directory
        if result_dir is None:
            self.result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
        else:
            self.result_dir = result_dir
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.responses_log: list[GPTResponse_metadata] = []
        self.reasoning_log: list[reasoning_and_web_search_response] = []
    

    def query_openai(
        self,
        filename: str,
        prompt: str,
        model: str = "gpt-5-2025-08-07",
        temperature: float = 1.0,
        max_output_tokens: int = 5000,
        reasoning_effort: str | None = "medium",
        verbosity: str = "medium",
        store: bool = False,
        safety_identifier: str | None = None,
        tools: list[dict] | None =  [{"type": "web_search"}],
        include: list[str] | None = ["web_search_call.action.sources"],
        save_raw_response: bool = True,
        web_search_options: dict | None = None
    ) -> GPTResponse_metadata:
        """
        Query GPT with optional reasoning and web search information extraction.

        Args:
            prompt: The user prompt to send to the model
            model: GPT model to use (default: "gpt-5-2025-08-07")
            temperature: Sampling temperature for response randomness (0.0-2.0, default: 1.0; GPT-5 can only take 1.0)
            max_output_tokens: Maximum tokens in the response (default: 5000)
            reasoning_effort: Level of reasoning detail - "minimal", "low", "medium", "high", or None (default: "medium")
            verbosity: Response verbosity level - "low", "medium", "high" (default: "medium")
            store: Whether to store the response for future retrieval (default: False)
            safety_identifier: Optional safety identifier for the request
            tools: List of tools to enable (default: [{"type": "web_search"}])
            include: List of additional data to include in response (default: ["web_search_call.action.sources"])
            web_search_options: Web search configuration including user_location and search_context_size
                Example: {
                    "search_context_size": "medium",
                    "user_location": {
                        "approximate": {
                            "city": "London",
                            "country": "GB",
                            "region": "England",
                            "timezone": "Europe/London"
                        }
                    }
                }

        Returns:
            GPTResponse_metadata: Object containing the response data, reasoning, and metadata

        Raises:
            Exception: If the API request fails or returns an error
        """
        
        actual_prompt = prompt
        
        # set up input parameters to LLMs
        try:
            # Build request parameters
            request_params = {
                "model": model,
                "input": [{"role": "user", "content": actual_prompt}],
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "store": store,
            }

            if reasoning_effort is not None and reasoning_effort in ["minimal", "low", "medium", "high"]:
                request_params["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}
            if verbosity is not None and verbosity in ["low", "medium", "high"]:
                request_params["text"] = {"verbosity": verbosity}
            if safety_identifier is not None:
                request_params["safety_identifier"] = safety_identifier

            # Build tools parameter with web_search_options merged in
            if tools is not None:
                # Merge web_search_options into the web_search tool
                merged_tools = []
                for tool in tools:
                    if tool.get("type") == "web_search" and web_search_options is not None:
                        # Merge the web_search_options into this tool
                        merged_tool = {"type": "web_search"}

                        # Add search_context_size if specified
                        if "search_context_size" in web_search_options:
                            merged_tool["search_context_size"] = web_search_options["search_context_size"]

                        # Add user_location if specified with validation
                        if "user_location" in web_search_options:
                            # Validate structure exists before accessing
                            if isinstance(web_search_options.get("user_location"), dict) and \
                               "approximate" in web_search_options["user_location"] and \
                               isinstance(web_search_options["user_location"]["approximate"], dict):
                                # Need to include the full structure with "type": "approximate"
                                user_loc = web_search_options["user_location"]["approximate"].copy()
                                user_loc["type"] = "approximate"
                                merged_tool["user_location"] = user_loc
                            else:
                                print("Warning: web_search_options['user_location']['approximate'] not properly formatted, skipping")

                        merged_tools.append(merged_tool)
                    else:
                        merged_tools.append(tool)

                request_params["tools"] = merged_tools

            if include is not None:
                request_params["include"] = include

            response = self.client.responses.create(**request_params)
            detailed_response = self.extract_reasoning_and_web_search(response)
            source_num = sum([i.get("web_search", {}).get("url_num", 0) for i in detailed_response if "web_search" in i.keys()])

            # Extract the ACTUAL web search options used from the response (not from input)
            actual_web_search_options = self.extract_web_search_options_from_response(response)

            # extract output responses
            gpt_response = GPTResponse_metadata(
                filename=filename,
                prompt=prompt,
                response=response.output_text,
                reasoning_num=self.count_reasoning_num(response),
                web_search_request_num=self.count_web_search_num(response),
                web_source_num=source_num,
                model=response.model,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                temperature=response.temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=response.reasoning.effort,
                verbosity=response.text.verbosity,
                store=response.store,
                safety_identifier=response.safety_identifier,
                tools=tools,
                include=include,
                prompt_tokens=response.usage.input_tokens,
                reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
                total_tokens=response.usage.total_tokens,
                id=response.id,
                web_search_options=actual_web_search_options
            )

            # extract reasoning and web search information
            reasoning_and_web_search = reasoning_and_web_search_response(
                filename=filename,
                prompt=prompt,
                detailed_response=detailed_response,
                id=response.id,
                web_search_options=actual_web_search_options
            )
            
            self.responses_log.append(gpt_response)
            self.reasoning_log.append(reasoning_and_web_search)

            if save_raw_response:
                response_handler = response_utils(result_dir=self.result_dir)
                response_handler.save_raw_response(response, filename)

            return response
            
        except Exception as e:
            print(f"Error querying {model}: {str(e)}")
            raise
    
    def count_reasoning_num(self, response):
        all_types = [t["type"] for t in response.to_dict()["output"]]
        return all_types.count("reasoning")

    def count_web_search_num(self, response):
        all_types = [t["type"] for t in response.to_dict()["output"]]
        return all_types.count("web_search_call")

    def extract_web_search_options_from_response(self, response):
        """
        Extract web search options from the response's tools field.
        This captures the ACTUAL location used by OpenAI, not what was requested.
        """
        response_dict = response.to_dict()
        tools = response_dict.get("tools", [])

        for tool in tools:
            if tool.get("type") == "web_search":
                web_search_options = {}

                # Extract search_context_size
                if "search_context_size" in tool:
                    web_search_options["search_context_size"] = tool["search_context_size"]

                # Extract user_location
                if "user_location" in tool:
                    user_loc = tool["user_location"]
                    # Build the approximate location object
                    approximate = {}

                    if user_loc.get("city"):
                        approximate["city"] = user_loc["city"]
                    if user_loc.get("country"):
                        approximate["country"] = user_loc["country"]
                    if user_loc.get("region"):
                        approximate["region"] = user_loc["region"]
                    if user_loc.get("timezone"):
                        approximate["timezone"] = user_loc["timezone"]

                    if approximate:
                        web_search_options["user_location"] = {"approximate": approximate}

                return web_search_options if web_search_options else None

        return None

    def extract_reasoning_and_web_search(self, response):
        """
        Extract reasoning and web search information from API response.
        
        Args:
            response: OpenAI API response object
            
        Returns:
            List of dicts containing reasoning, web_search, and message data
        """
        result = []
        for i in response.to_dict()["output"]:
            if i["type"] == "reasoning":
                # Validate summary structure before accessing
                if "summary" in i and isinstance(i["summary"], list):
                    if len(i["summary"]) == 1:
                        # Single summary item - try to extract text
                        summary_item = i["summary"][0]
                        if isinstance(summary_item, dict) and "text" in summary_item:
                            result.append({"reasoning": summary_item["text"]})
                        else:
                            # Fallback: use the whole summary item
                            result.append({"reasoning": i["summary"]})
                    else:
                        # Multiple summary items or empty list
                        result.append({"reasoning": i["summary"]})
                else:
                    # No summary or invalid format
                    result.append({"reasoning": "N/A"})
                    
            elif i["type"] == "web_search_call":
                # Check if the web_search_call has an action field
                if "action" in i and "query" in i["action"]:
                    sources = i["action"].get("sources", [])
                    result.append({
                        "web_search": {
                            "url_num": len(sources),
                            "query": i["action"]["query"],
                            "sources": sources
                        }
                    })
                else:
                    # Handle web_search_call without action (e.g., status-only entries)
                    result.append({
                        "web_search": {
                            "url_num": 0,
                            "query": "N/A",
                            "sources": []
                        }
                    })
                    
            elif i["type"] == "message":
                # Validate content structure before accessing
                if "content" in i and isinstance(i["content"], list) and len(i["content"]) > 0:
                    content_item = i["content"][0]
                    if isinstance(content_item, dict):
                        annotations = content_item.get("annotations", [])
                        text = content_item.get("text", "")
                        result.append({
                            "message": {
                                "url_num": len(annotations),
                                "url_annotations": annotations,
                                "content": text
                            }
                        })
                    else:
                        # Invalid content item format
                        result.append({
                            "message": {
                                "url_num": 0,
                                "url_annotations": [],
                                "content": str(content_item)
                            }
                        })
                else:
                    # No content or empty content list
                    result.append({
                        "message": {
                            "url_num": 0,
                            "url_annotations": [],
                            "content": "N/A"
                        }
                    })
        return result
        

def main():
    pass

if __name__ == "__main__":
    main()
