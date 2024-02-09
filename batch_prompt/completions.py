from math import ceil
from time import time
from tqdm import tqdm, trange
from pprint import pprint

import aiohttp
from tenacity import retry, stop_after_attempt, wait_random_exponential

from batch_prompt.utils import openai, print_call_summary


DEFAULT_MODEL_ARGS = {
    'engine': 'gpt-3.5-turbo-instruct',
    'n': 1,
    'max_tokens': 5,
    'temperature': 1.0,
}


@retry(wait=wait_random_exponential(min=1, max=70), stop=stop_after_attempt(15))
def complete_backoff(*args, **kwargs):
    """Retry + backoff to handle timeout errors, and other noisy errors like 502 bad gateway
          https://platform.openai.com/docs/guides/rate-limits/error-mitigation"""
    return openai.Completion.create(*args, **kwargs)


def single_complete(formatted_prompts, prompts, prompt_args, m_args, model_args):
    completion = complete_backoff(prompt=formatted_prompts, **m_args)

    n = m_args.get('n', 1) 
    results = [
        {
            'prompt': p,
            'choice': c,  # only 1 choice from the completion corresponds to this prompt data point
            'completion': completion,
            'prompt_raw': prompts[p_i],
            'prompt_args': prompt_args[p_i],
            'model_args': model_args,
        }
        for p_i, p in enumerate(formatted_prompts)
        for c in completion.choices[n*p_i : n*(p_i+1)]
    ]
    return results, completion


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


def get_completions(prompt, prompt_args=None, model_args=None, verbose=2, queries_per_batch=5):
    
    prompts, prompt_args = listify_prompts(prompt, prompt_args)
    formatted_prompts = [p.format(**kwargs) for p, kwargs in zip(prompts, prompt_args)]
    
    # Model Args
    m_args = DEFAULT_MODEL_ARGS.copy()
    if model_args is not None:
        m_args.update(model_args)

    if verbose > 1:
        print('='*80)
        print('Calling OpenAI API . . .')
        t1 = time()
    if verbose > 2:
        print('------- Prompts -------')
        pprint(formatted_prompts)
        print('------- API Args -------')
        pprint(m_args)
    
    # Split queries into batches and call OpenAI API
    num_batches = ceil(len(prompt_args) / queries_per_batch)

    range_ = trange if (verbose > 0 and num_batches > 1) else range
    nq = ceil(len(prompt_args) / num_batches)

    completions = []
    results = []

    for b in range_(num_batches):
        i1, i2 = b*nq, (b+1)*nq

        res, completion = single_complete(
            formatted_prompts[i1:i2], prompts[i1:i2], prompt_args[i1:i2], 
            m_args, model_args)

        results += res
        completions.append(completion)
    
    # Return list of formatted dictionaries
    if verbose > 1:
        print_call_summary(t1, len(results), completions)

    return results
