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

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from together import Together, AsyncTogether

# from batch_prompt import keys
from batch_prompt import keys_eb as keys   # TODO


# This fixes a weird issue with FAS-RC where OpenAI queries at `httpx.get(..)` cause SSL credentials errors
#   Minimal SLL error test: `import httpx; httpx.get('https://httpbin.org/get')`
#   FAS-RC docs on proxy, related but didn't directly help me: docs.rc.fas.harvard.edu/kb/proxy-settings/
http_client = httpx.Client(verify=requests.certs.where())
http_async  = httpx.AsyncClient(verify=requests.certs.where())


# Map of clients for each service
CLIENTS = {

    # https://platform.openai.com/docs/api-reference
    'openai': {
        'sync': OpenAI(
            api_key=keys.OPENAI_API_KEY,
            organization=keys.OPENAI_ORGANIZATION,
            http_client=http_client
        ),
        'async': AsyncOpenAI(
            api_key=keys.OPENAI_API_KEY,
            organization=keys.OPENAI_ORGANIZATION,
            http_client=http_async
        )
    } if keys.OPENAI_API_KEY != 'MY_API_KEY' else None,

    # https://github.com/Azure-Samples/openai
    'azure': {
        'sync': AzureOpenAI(
            api_key=keys.AZURE_API_KEY,
            azure_endpoint=keys.AZURE_OPENAI_ENDPOINT,
            api_version='2024-02-01',   #'2022-12-01',
            http_client=http_client
        ),
        'async': AsyncAzureOpenAI(
            api_key=keys.AZURE_API_KEY,
            azure_endpoint=keys.AZURE_OPENAI_ENDPOINT,
            api_version='2024-02-01',   #'2022-12-01',
            http_client=http_async
        )
    } if keys.OPENAI_API_KEY != 'MY_API_KEY' else None,

    # https://docs.together.ai/reference/completions
    'together': {
        'sync': Together(
            api_key=keys.TOGETHER_API_KEY,
        ),
        'async': AsyncTogether(
            api_key=keys.TOGETHER_API_KEY,
        )
    } if keys.TOGETHER_API_KEY != 'MY_API_KEY' else None,

}


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


def run_async(call_fn, inputs_ls, model_args, verbose=1, queries_per_batch=1, 
              backend='openai', concurrency=1000, is_chat=False):
    qpb = queries_per_batch
    n_inputs = len(inputs_ls)
    concurrency = min(concurrency, n_inputs)

    async def f(n1, n2):
        n2 = min(n2, n_inputs)
        async_calls = [asyncio.create_task(call_fn(backend)(
                        inputs_ls[i : i+qpb] if qpb > 1 else inputs_ls[i], **model_args)) 
                       for i in np.arange(n1, n2, qpb)]

        gather = asyncio.gather if verbose == 0 else tqdm_asyncio.gather
        return await gather(*async_calls)

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
retry = retry_tenacity(wait=wait_random_exponential(exp_base=1.2, multiplier=.5, max=70), 
                       stop=stop_after_attempt(300))
# retry = retry_tenacity(wait=wait_random(), stop=stop_after_attempt(100))
# retry = lambda x: x          # Dummy retry func, useful for debugging 
