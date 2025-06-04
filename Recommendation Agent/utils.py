import re
import json

def extract_json_from_llm_response(text):
    """
    Extracts JSON from an LLM response that may include code block formatting like ```json ... ```
    Returns the parsed JSON as a Python dict, or None if no valid JSON is found.
    """
    # Match content inside triple backticks, optionally labeled as ```json
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # Fallback: try to find any JSON-like object or array
        code_fallback = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
        if not code_fallback:
            return None
        json_str = code_fallback.group(1)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
