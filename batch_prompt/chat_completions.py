from tqdm.asyncio import tqdm_asyncio
from time import time
from pprint import pprint

import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential

from batch_prompt.utils import openai, print_call_summary


DEFAULT_MODEL_ARGS = {
    'model': 'gpt-3.5-turbo',
    'n': 1,
    'max_tokens': 5,
    'temperature': 1.0,
}


@retry(wait=wait_random_exponential(min=1, max=70), stop=stop_after_attempt(15))
async def chat_async_backoff(*args, **kwargs):
    """Retry + backoff to handle timeout errors, and other noisy errors like 502 bad gateway
          https://platform.openai.com/docs/guides/rate-limits/error-mitigation"""
    return await openai.ChatCompletion.acreate(*args, **kwargs)


def complete_chat_async(messages_ls, model_args, verbose=1):
    async def f():
        async with aiohttp.ClientSession() as session:
            openai.aiosession.set(session)

            async_calls = [asyncio.ensure_future(chat_async_backoff(messages=messages, **model_args)) 
                           for messages in messages_ls]

            gather = asyncio.gather if verbose == 0 else tqdm_asyncio.gather
            return await gather(*async_calls)

    # Call OpenAI async
    completions = asyncio.run(f())
    return completions
    

def flatten(ls):
    return [i for subl in ls for i in subl]

def listify_messages(messages, messages_args=None):
    if len(messages) > 0 and type(messages[0]) is list: 
        return flatten([listify_messages(msgs, messages_args) for msgs in messages])
    if messages_args and type(messages_args[0]) is list:
        return flatten([listify_messages(messages, msg_args) for msg_args in messages_args])

    messages_args = messages_args or [{} for _ in messages]
    return [(messages, messages_args)]



def get_chat_completions(messages, messages_args=None, model_args=None, verbose=2):
    messages, messages_args = zip(*listify_messages(messages, messages_args))
    formatted_msgs = [[{'role': m['role'], 
                        'content': m['content'].format(**kwargs)}
                       for m, kwargs in zip(msgs, msg_args)]
                      for msgs, msg_args in zip(messages, messages_args)]

    # Model Args
    m_args = DEFAULT_MODEL_ARGS.copy()
    if model_args is not None:
        m_args.update(model_args)
    n = m_args.get('n', 1)

    if verbose > 1:
        print('='*80)
        print('Calling OpenAI Chat API . . .')
        t1 = time()
    if verbose > 2:
        print('------- Messages -------')
        pprint(formatted_msgs)

    completions = complete_chat_async(formatted_msgs, m_args, verbose)

    # Return list of formatted dictionaries
    results = [
        {
            'choice': c,
            'completion': completion,
            'messages': formatted_msgs[idx],
            'messages_raw': messages[idx],
            'messages_args': messages_args[idx],
            'model_args': model_args,
        }
        for idx, completion in enumerate(completions)
        for c in completion.choices
    ]
    if verbose > 1:
        print_call_summary(t1, len(results), completions)

    return results

