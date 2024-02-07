from time import time
from tqdm import tqdm, trange
from pprint import pprint

import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils import openai, listify_prompts


DEFAULT_MODEL_ARGS = {
    'model': 'gpt-3.5-turbo',
    'n': 1,
    'max_tokens': 5,
    'temperature': 1.0,
}


# Retry + backoff to handle timeout errors, and other noisy errors like 502 bad gateway
#   https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@retry(wait=wait_random_exponential(min=1, max=70), stop=stop_after_attempt(50))
async def chat_async_backoff(*args, **kwargs):
    return await openai.ChatCompletion.acreate(*args, **kwargs)

async def get_completion(prompt_instruct, prompt_context, prompt_system, model_args):
    """Format 3 prompts into a single set of messages. 


    TODO: could this be modular, to support a list of messages instead of doing this here?
    """
    chat_completion = await chat_async_backoff(
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_instruct},
            {"role": "assistant", "content": prompt_context},
        ],
        **model_args
    )
    return chat_completion


def call_chat_async(prompts_instruct, prompts_context, system_prompt, model_args):
    async def f():
        async with aiohttp.ClientSession() as session:
            openai.aiosession.set(session)

            async_calls = [asyncio.ensure_future(get_completion(pi, pc, system_prompt, model_args)) 
                           for pi, pc in zip(prompts_instruct, prompts_context)]

            results = await asyncio.gather(*async_calls)
            return results

    # Call OpenAI async
    completions = asyncio.run(f())
    return completions
    

def get_chat_completions(instructs, contexts, system_prompt='You are a helpful assistant.',
                         instruct_args=None, context_args=None, model_args=None, verbose=1):

    ### TODO: batch
    ### TODO: can this be an abstract fn for both completions + chat APIs?
    
    instructs, instruct_args = listify_prompts(instructs, instruct_args)
    formatted_instructs = [p.format(**kwargs) for p, kwargs in zip(instructs, instruct_args)]

    contexts, context_args = listify_prompts(contexts, context_args)
    formatted_contexts = [p.format(**kwargs) for p, kwargs in zip(contexts, context_args)]

    instruct_args, context_args = zip(*[(ia, xa) for ia in instruct_args for xa in context_args])
    formatted_instructs, formatted_contexts = zip(*[(i, x) for i in formatted_instructs for x in formatted_contexts])
    
    # Model Args
    m_args = DEFAULT_MODEL_ARGS.copy()
    if model_args is not None:
        m_args.update(model_args)
    n = m_args.get('n', 1) 

    if verbose > 0:
        print('='*80)
        print('Calling OpenAI Chat API . . .')
        t1 = time()
    if verbose > 1:
        print('------- Instruct + Context Prompts -------')
        pprint(list(zip(formatted_instructs, formatted_contexts)))

    completions = call_chat_async(formatted_instructs, formatted_contexts, system_prompt, model_args)
    
    # Return list of formatted dictionaries
    results = [
        {
            'instruct': i,
            'context': x,
            'choice': c,  # only 1 choice from the completion corresponds to this prompt data point
            'completion': completion,
            'instruct_raw': formatted_instructs[idx],
            'instruct_args': instruct_args[idx],
            'context_raw': formatted_contexts[idx],
            'context_args': context_args[idx],
            'model_args': model_args,
        }
        for completion in completions
        for idx, (i, x) in enumerate(zip(formatted_instructs, formatted_contexts))
        for c in completion.choices[n*idx : n*(idx+1)]
        # for p_i, p in enumerate(formatted_prompts)
    ]
    if verbose > 0:
        t2 = time()
        print('~'*15, 'Done', '~'*15)
        print(f'Time: {t2 - t1 :.2f}s')
        print(f'Number of results: {len(results)}')
        pprint([dict(completion['usage']) for completion in completions])

    return results





if __name__ == '__main__':
    ##### TODO: simple example in main
    #####
    #####

    pass