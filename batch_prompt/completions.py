from time import time
from tqdm import tqdm, trange
from pprint import pprint

import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils import openai, listify_prompts


DEFAULT_MODEL_ARGS = {
    'engine': 'gpt-3.5-turbo-instruct',
    'n': 1,
    'max_tokens': 5,
    'temperature': 1.0,
}


# Retry + backoff to handle timeout errors, and other noisy errors like 502 bad gateway
#   https://platform.openai.com/docs/guides/rate-limits/error-mitigation
@retry(wait=wait_random_exponential(min=1, max=70), stop=stop_after_attempt(50))
def completion_backoff(*args, **kwargs):
    return openai.Completion.create(*args, **kwargs)


def _call_single(formatted_prompts, prompts, prompt_args, m_args, model_args):
    completion = completion_backoff(prompt=formatted_prompts, **m_args)

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

def get_completions(prompt, prompt_args=None, model_args=None, verbose=1, num_batches=1):
    
    prompts, prompt_args = listify_prompts(prompt, prompt_args)
    formatted_prompts = [p.format(**kwargs) for p, kwargs in zip(prompts, prompt_args)]
    
    # Model Args
    m_args = DEFAULT_MODEL_ARGS.copy()
    if model_args is not None:
        m_args.update(model_args)

    if verbose > 0:
        print('='*80)
        print('Calling OpenAI . . .')
        t1 = time()
    if verbose > 1:
        print('------- Prompts -------')
        pprint(formatted_prompts)
        print('------- API Args -------')
        pprint(m_args)
    
    # Split queries into batches and call OpenAI API
    range_ = trange if (verbose != 0 and num_batches > 1) else range
    nb = (len(prompt_args) // num_batches) + 1

    completions = []
    results = []

    for b in range_(num_batches):
        i1, i2 = b*nb, (b+1)*nb
        res, completion = _call_single(
            formatted_prompts[i1:i2], prompts[i1:i2], prompt_args[i1:i2], 
            m_args, model_args)

        results += res
        completions.append(completion)
    
    # Return list of formatted dictionaries
    if verbose > 0:
        t2 = time()
        print('~'*15, 'Done', '~'*15)
        print(f'Time: {t2 - t1 :.2f}s')
        print(f'Number of results: {len(results)}')

        # Aggregate usage tokens across batches
        usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
        for completion in completions:
            for k, v in completion['usage'].items():
                usage[k] += v
        pprint(usage)

    return results





if __name__ == '__main__':
    ##### TODO: simple example in main
    #####
    #####


    # p = """Q: Are the following coin flips from a random coin flip, or non-random coin flip? Why? [Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails]
    #
    # A: The flips are from a"""
    # res = get_completions(p, {'max_tokens': 1})

    n_samples = 64
    max_tokens = 300
    time_sleep = 70

    # TOKEN_LIMIT = 9000
    # n_calls = math.ceil((n_samples * max_tokens) / TOKEN_LIMIT)


    # res = []

    # for p_tails in tqdm([10, 30, 50, 70, 90]):
    #     p_heads = 100 - p_tails
    #     source_coin = 'a weighted coin, with {}% probability of Heads and {}% probability of Tails'
    #     source_fair = 'a fair coin, with 50% probability of Heads and 50% probability of Tails'

    #     prompt_instruct = 'Generate a sequence of 1000 random samples from {source}.'
    #     prompt_context = '[{flips}'

    #     # ---------------------------------------------------------------------------
    #     for _ in range(2):
    #         res_ = call_openai_chat_async(
    #             prompt_instruct, 
    #             prompt_context,
    #             system_prompt='Your responses will only consist of comma-separated "Heads" and "Tails" samples.' + \
    #                           '\nDo not repeat the user\'s messages in your responses.',
    #             instruct_args={'source': source_coin.format(p_heads, p_tails) if p_heads != 50 else source_fair},
    #             context_args={'flips': 'Heads,'},
    #             model_args={'max_tokens': max_tokens, 'n': n_samples, 'model': 'gpt-4'})

