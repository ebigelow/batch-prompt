import openai
import keys

openai.organization = keys.ORGANIZATION
openai.api_key = keys.API_KEY


def listify_prompts(prompt, prompt_args=None):        
    """
    Format prompt str with prompt args, and return (<list of prompts>, <list of kwargs for each>)
    
    Args:
        prompt (str | list[str]) : can be one prompt (str) or multiple prompts (list[str])
        prompt (dict | list[dict]) : can be one kwarg dict or a list of kwarg dicts
    """
    # Prompt args
    prompt_args = prompt_args or {}   # default value is empty dict
    prompt_args = prompt_args if type(prompt_args) is list else [prompt_args]  # convert to list
        
    # Format prompt / prompts
    if type(prompt) is str:
        prompts = [prompt for kwargs in prompt_args]
    else:
        prompts = [p for p in prompt for kwargs in prompt_args]
        prompt_args = [kwargs for p in prompt for kwargs in prompt_args]
        
    return prompts, prompt_args
