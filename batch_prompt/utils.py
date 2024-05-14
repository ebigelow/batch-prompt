import sys
from collections import defaultdict
from time import time
from pprint import pprint
import numpy as np

import requests
import httpx
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tenacity import stop_after_delay, wait_random_exponential, retry as retry_tenacity

from openai import OpenAI, AsyncOpenAI

from batch_prompt import keys_eb as keys

if keys.API_KEY == "MY_API_KEY":
    raise ImportError('Add your OpenAI API key to keys.py')

client = OpenAI(
    api_key=keys.API_KEY,
    organization=keys.ORGANIZATION,
    http_client=httpx.Client(verify=requests.certs.where())
)
client_async = AsyncOpenAI(
    api_key=keys.API_KEY,
    organization=keys.ORGANIZATION,
    http_client=httpx.AsyncClient(verify=requests.certs.where())
)

USE_ASYNC = not ("ipykernel" in sys.modules)   # default: use async unless we're in jupyter


def simplify_completion(completion):
    return {k: v for k, v in completion.dict().items() if k != 'choices'}


def print_call_summary(t1, n_res, completions):
    """Helper method to print summary stats and query useage for a batch of calls."""
    t2 = time()
    print('~'*15, 'Done', '~'*15)
    print(f'Time: {t2 - t1 :.2f}s')
    print(f'Number of results: {n_res}')

    # Aggregate usage tokens across batches
    usage = defaultdict(lambda: 0)
    for completion in completions:
        for k, v in completion.dict()['usage'].items():
            usage[k] += v
    pprint(dict(usage))


def run_async(call_fn, inputs_ls, model_args, verbose=1, queries_per_batch=1, is_chat=False):
    qpb = queries_per_batch

    async def f():
        async_calls = [asyncio.ensure_future(call_fn(
                        inputs_ls[i : i+qpb] if qpb > 1 else inputs_ls[i], **model_args)) 
                       for i in np.arange(0, len(inputs_ls), qpb)]

        gather = asyncio.gather if verbose == 0 else tqdm_asyncio.gather
        return await gather(*async_calls)

    # Call OpenAI async
    completions = asyncio.run(f())
    return completions


# Retry + backoff to handle timeout errors, and other noisy errors like 502 bad gateway
#          https://platform.openai.com/docs/guides/rate-limits/error-mitigation"""

retry = retry_tenacity(wait=wait_random_exponential(min=1, max=70))   # , stop=stop_after_attempt(100)
# retry = lambda x: x          # Dummy retry func, useful for debugging 
