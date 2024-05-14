import sys
from collections import defaultdict
from time import time
from pprint import pprint
import numpy as np

import requests
import httpx
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tenacity import stop_after_attempt, wait_random_exponential, wait_random, retry as retry_tenacity

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
        for k, v in completion.usage.dict().items():
            usage[k] += v
    pprint(dict(usage))


async def gather_with_concurrency(gather, n, *coros):
    """
    Limit the amount of concurrent calls from asyncio.

    From: https://stackoverflow.com/a/61478547/4248948
    """
    semaphore = asyncio.Semaphore(n)
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await gather(*(sem_coro(c) for c in coros))

def run_async(call_fn, inputs_ls, model_args, verbose=1, queries_per_batch=1, 
              concurrency=100, is_chat=False):
    qpb = queries_per_batch
    n_inputs = len(inputs_ls)
    concurrency = min(concurrency, n_inputs)

    async def f(n1, n2):
        n2 = min(n2, n_inputs)
        async_calls = [asyncio.create_task(call_fn(
                        inputs_ls[i : i+qpb] if qpb > 1 else inputs_ls[i], **model_args)) 
                       for i in np.arange(n1, n2, qpb)]

        gather = asyncio.gather if verbose == 0 else tqdm_asyncio.gather
        return await gather_with_concurrency(gather, concurrency, *async_calls)
        # return await gather(*async_calls)

    range_ = np.arange(0, n_inputs, concurrency)
    range_ = tqdm(range_) if verbose else range_

    # Call OpenAI in batches of async calls
    #   Note: without this batching, the loop above hangs for me with concurrency >2000
    completions = []
    for n1 in range_:
        completions += asyncio.run(f(n1, n1 + concurrency))
    return completions


# Retry + backoff to handle timeout errors, and other noisy errors like 502 bad gateway
#          https://platform.openai.com/docs/guides/rate-limits/error-mitigation"""

# retry = retry_tenacity(wait=wait_random_exponential(exp_base=1.1, multiplier=.5, max=70), stop=stop_after_attempt(100))
retry = retry_tenacity(wait=wait_random(), stop=stop_after_attempt(100))
# retry = lambda x: x          # Dummy retry func, useful for debugging 
