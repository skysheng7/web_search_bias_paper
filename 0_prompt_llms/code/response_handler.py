"""
Utility functions for LLM Bias Analysis
"""
import os
import json
import dataclasses
from dataclasses import dataclass

class response_utils:
    """
    Utility functions for responses
    """

    def __init__(self, result_dir: str = None):
        if result_dir is None:
            self.result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
        else:
            self.result_dir = result_dir
        # Create result directory if it doesn't exist
        os.makedirs(self.result_dir, exist_ok=True)

    def save_responses_log(self, responses_log: list, filename: str = "gpt_responses.json"):
        """
        Save all logged responses to a JSON file for analysis.
        
        If the file exists, loads it and appends new data. If the file doesn't exist
        or is corrupted, creates a new file with the new data.
        
        Args:
            responses_log: List of response objects (dataclasses) to save
            filename: Name of the JSON file (without .json extension)
        """
        filepath = os.path.join(self.result_dir, f"{filename}.json")
        
        # Convert dataclasses to dicts for JSON serialization
        new_data = [dataclasses.asdict(response) for response in responses_log]
        
        if os.path.exists(filepath):
            try:
                # Try to load existing JSON file
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Ensure existing_data is a list
                if not isinstance(existing_data, list):
                    print(f"Warning: Existing {filepath} is not a list, will overwrite")
                    existing_data = []
                
                # Extend with new data
                existing_data.extend(new_data)
                new_data = existing_data
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse existing {filepath}: {e}")
                print("Will overwrite with new data")
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")
                print("Will overwrite with new data")
        
        # Write the combined data (or just new data if file didn't exist)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        print(f"Responses saved to: {filepath}")
    
    def save_raw_response(self, response, filename: str):

        if filename is not None:
            path = os.path.join(self.result_dir, f"{filename}.json")
            with open(path, "w", encoding='utf-8') as f:
                # Parse the JSON string first, then dump with formatting
                response_data = json.loads(response.to_json())
                json.dump(response_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError("Filename is required")
    
    def save_reasoning_markdown(self, reasoning_log: list, filename: str = "gpt_reasoning"):
        """
        Save reasoning log in simple markdown format.
        
        Note: This function APPENDS to the markdown file if it exists. This allows
        building up a cumulative log across multiple batch runs. If you want to
        start fresh, delete the existing .md file before running.
        
        Args:
            reasoning_log: List of reasoning_and_web_search_response objects
            filename: Name for the output markdown file (without .md extension)
        """
        if not reasoning_log:
            print("No reasoning log data to export")
            return
            
        filepath = os.path.join(self.result_dir, f"{filename}.md")
        
        # Open in append mode - creates file if it doesn't exist, appends if it does
        with open(filepath, 'a', encoding='utf-8') as f:
            for entry in reasoning_log:
                f.write(f"# Prompt: {entry.prompt}\n\n")
                f.write(f"1. Filename: {entry.filename}\n\n")
                f.write(f"2. ID: {entry.id}\n\n")

                # Add web search location info if available
                if hasattr(entry, 'web_search_options') and entry.web_search_options:
                    f.write("3. **Web Search Location:**\n\n")
                    if "user_location" in entry.web_search_options:
                        loc = entry.web_search_options["user_location"].get("approximate", {})
                        location_parts = []
                        if "city" in loc:
                            location_parts.append(f"City: {loc['city']}")
                        if "country" in loc:
                            location_parts.append(f"Country: {loc['country']}")
                        if "region" in loc:
                            location_parts.append(f"Region: {loc['region']}")
                        if "timezone" in loc:
                            location_parts.append(f"Timezone: {loc['timezone']}")

                        if location_parts:
                            f.write("   - " + ", ".join(location_parts) + "\n\n")
                        else:
                            f.write("   - Not specified\n\n")

                    if "search_context_size" in entry.web_search_options:
                        f.write(f"   - Search Context Size: {entry.web_search_options['search_context_size']}\n\n")
                    f.write("\n")

                f.write("## Detailed Response:\n\n")
                
                for component in entry.detailed_response:
                    if "reasoning" in component:
                        reasoning_text = component["reasoning"]
                        if len(reasoning_text) > 0:
                            f.write("**🧠 REASONING:**\n\n")
                        
                            if isinstance(reasoning_text, list):
                                for reason in reasoning_text:
                                    if isinstance(reason, dict) and "text" in reason:
                                        f.write(f"{reason['text']}\n\n")
                                    else:
                                        f.write(f"{reason}\n\n")
                            else:
                                f.write(f"{reasoning_text}\n\n")
                            f.write("\n")
                    
                    elif "web_search" in component:
                        search_data = component["web_search"]
                        f.write("**🌐 WEB SEARCH:**\n\n")
                        f.write("\n")
                        f.write(f"Query: {search_data['query']}\n\n")
                        f.write("\n")
                        f.write(f"Number of sources: {search_data['url_num']}\n\n")
                        f.write("\n")
                        
                        if search_data.get("sources"):
                            f.write("*Sources:*\n\n")
                            for source in search_data["sources"]:
                                if isinstance(source, dict):
                                    url = source.get("url", "#")
                                    f.write(f"- {url}\n\n")
                                    f.write("\n")
                                else:
                                    f.write(f"- {source}\n\n")
                                    f.write("\n")
                        f.write("\n")
                    
                    elif "message" in component:
                        # Handle message component - extract content and annotations
                        message_data = component["message"]
                        
                        if isinstance(message_data, dict):
                            # Extract content (can be string or list)
                            content = message_data.get("content", "")
                            
                            # Write the output message
                            f.write("**💭 OUTPUT MESSAGE:**\n\n")
                            f.write("\n")
                            if isinstance(content, str):
                                f.write(f"{content}\n\n")
                            elif isinstance(content, list):
                                # Handle list of content items
                                for content_item in content:
                                    if isinstance(content_item, dict) and "text" in content_item:
                                        f.write(f"{content_item['text']}\n\n")
                                    else:
                                        f.write(f"{content_item}\n\n")
                            f.write("\n")
                            
                            # Extract annotations - check both possible keys
                            annotations = message_data.get("url_annotations") or message_data.get("annotations") or []
                            
                            # If content is a list, also check for annotations within content items
                            if isinstance(content, list) and not annotations:
                                for content_item in content:
                                    if isinstance(content_item, dict) and "annotations" in content_item:
                                        annotations = content_item.get("annotations", [])
                                        break
                            
                            # Render annotations if they exist
                            if annotations and len(annotations) > 0:
                                f.write("**REFERENCED SOURCES:**\n\n")
                                f.write("\n")
                                for annotation in annotations:
                                    if isinstance(annotation, dict):
                                        title = annotation.get('title', 'No Title')
                                        url = annotation.get('url', '#')
                                        f.write(f"- [{title}]({url})\n\n")
                                    else:
                                        f.write(f"- {annotation}\n\n")
                                f.write("\n")
                        else:
                            # Fallback for unexpected structure
                            f.write("**💭 OUTPUT MESSAGE:**\n\n")
                            f.write("\n")
                            f.write(f"{message_data}\n\n")
                            f.write("\n")
                
                f.write("\n")
                f.write("=" * 80 + "\n\n")
        
        print(f"Reasoning analysis saved to: {filepath}")
    
    
# dataclass to store reasoning & web search information
@dataclass
class reasoning_and_web_search_response:
    """
    Structure to hold reasoning and web search information
    """
    filename: str
    prompt: str
    detailed_response: list
    id: str
    web_search_options: dict | None = None