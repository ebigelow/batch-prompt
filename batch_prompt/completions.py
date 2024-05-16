from math import ceil
from time import time
from tqdm import tqdm, trange
from pprint import pprint
from wrapt_timeout_decorator import timeout

from batch_prompt.utils import retry, print_call_summary, run_async, CLIENTS, simplify_completion


DEFAULT_MODEL_ARGS = {
    'model': 'gpt-3.5-turbo-instruct-0914',
    'n': 1,
    'max_tokens': 5,
    'temperature': 1.0,
}


@retry
def complete_backoff(backend, *args, **kwargs):
    client = CLIENTS[backend]['sync']
    return client.completions.create(*args, **kwargs)

def complete_async_backoff(backend):
    @retry
    @timeout(20)
    async def f(prompt, *args, **kwargs):
        client = CLIENTS[backend]['async']
        return await client.completions.create(prompt=prompt, *args, **kwargs)
    return f

def batch_acomplete(formatted_prompts, prompts, prompt_args, m_args, model_args, backend, verbose, queries_per_batch, **kwargs):
    completions = run_async(complete_async_backoff, formatted_prompts, m_args, verbose, queries_per_batch, 
                            backend=backend, is_chat=False, **kwargs)

    n = m_args.get('n', 1) 
    qpb = lambda p_i: p_i % queries_per_batch

    results = [
        {
            'prompt': p,
            'choice': c.dict(),   # convert to dict for backward compatility
            'completion': simplify_completion(completions[p_i // queries_per_batch]),
            'prompt_raw': prompts[p_i],
            'prompt_args': prompt_args[p_i],
            'model_args': model_args,
        }
        for p_i, p in enumerate(formatted_prompts)    
        for c in completions[p_i // queries_per_batch].choices[n*qpb(p_i) : n*(qpb(p_i) + 1)]   # un-batch queries
    ]
    return results, completions

def single_complete(formatted_prompts, prompts, prompt_args, m_args, model_args, backend):
    completion = complete_backoff(backend, prompt=formatted_prompts, **m_args)

    n = m_args.get('n', 1) 
    results = [
        {
            'prompt': p,
            'choice': c.dict(),  # only 1 choice from the completion corresponds to this prompt data point
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
    prompt_args = prompt_args if type(prompt_args) in (list, tuple) else [prompt_args]  # convert to list
        
    # Format prompt / prompts
    if type(prompt) is str:
        prompts = [prompt for kwargs in prompt_args]
    else:
        prompts = [p for p in prompt for kwargs in prompt_args]
        prompt_args = [kwargs for p in prompt for kwargs in prompt_args]
        
    return prompts, prompt_args


def get_completions(prompt, prompt_args=None, model_args=None, verbose=2, 
                    queries_per_batch=1, use_async=True, backend='openai'):
    """
    Query backend completions API, auto-formatting `prompt`s with `prompt_arg`s.

    Arguments:
        prompt (str | list[str]): Prompt or list of prompts.
        prompt_args (dict, list[dict]): Prompt args dict (or list of these) with args for
            string formatting of `prompt`.
        model_args (dict): Args for backend API
        verbose (int[0-3]): How verbose will printing be
        queries_per_batch (int): How many prompts will go in each API batch? 
             Only works with OpenAI backend
        use_async (bool): Use async or sync. Set to True when querying in notebooks.
        backend (str): Which backend to use ('openai' | 'azure' | 'together')
    """
    
    prompts, prompt_args = listify_prompts(prompt, prompt_args)
    formatted_prompts = [p.format(**kwargs) for p, kwargs in zip(prompts, prompt_args)]
    
    # Model Args
    m_args = DEFAULT_MODEL_ARGS.copy()
    if model_args is not None:
        m_args.update(model_args)

    if verbose > 1:
        print('='*80)
        print(f'Calling {backend} API  . . .    (async={use_async})')
        t1 = time()
    if verbose > 2:
        print('------- Prompts -------')
        pprint(formatted_prompts)
        print('------- API Args -------')
        pprint(m_args)

    if use_async:
        results, completions = batch_acomplete(formatted_prompts, prompts, prompt_args, m_args, model_args,
                                               backend, verbose, queries_per_batch)
    else:
        # Split queries into batches and call backend API
        num_batches = ceil(len(prompt_args) / queries_per_batch)

        range_ = trange if (verbose > 0 and num_batches > 1) else range
        nq = ceil(len(prompt_args) / num_batches)

        completions = []
        results = []

        for b in range_(num_batches):
            i1, i2 = b*nq, (b+1)*nq

            res, completion = single_complete(
                formatted_prompts[i1:i2], prompts[i1:i2], prompt_args[i1:i2], 
                m_args, model_args, backend)

            results += res
            completions.append(completion)
    
    # Return list of formatted dictionaries
    if verbose > 1:
        print_call_summary(t1, len(results), completions)

    return results


