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
from wrapt_timeout_decorator import timeout

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from together import Together, AsyncTogether

# from batch_prompt import keys
from batch_prompt import keys_eb as keys   # TODO


# This fixes a weird issue with FAS-RC where OpenAI queries at `httpx.get(..)` cause SSL credentials errors
#   Minimal SLL error test: `import httpx; httpx.get('https://httpbin.org/get')`
#   FAS-RC docs on proxy, related but didn't directly help me: docs.rc.fas.harvard.edu/kb/proxy-settings/
http_client = httpx.Client(verify=requests.certs.where())
http_async  = httpx.AsyncClient(verify=requests.certs.where())

# Get google auth credentials
#   https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library#call-chat-completions-api
import vertexai
from google.auth import default, transport
vertexai.init(project=keys.GOOGLE_PROJECT, location=keys.GOOGLE_LOCATION)
try:
    GOOGLE_CREDENTIALS, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    GOOGLE_CREDENTIALS.refresh(transport.requests.Request())
except Exception as e:
    print('Warning: google credentials not loaded\n', e)
    GOOGLE_CREDENTIALS = None

# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters
GOOGLE_SAFETY = {
    'safety_settings': [
        {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
         'threshold': 'BLOCK_ONLY_HIGH'},    # BLOCK_NONE
        {'category': 'HARM_CATEGORY_HATE_SPEECH',
         'threshold': 'BLOCK_ONLY_HIGH'},
        {'category': 'HARM_CATEGORY_HARASSMENT',
         'threshold': 'BLOCK_ONLY_HIGH'},
        {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',    # https://www.python-httpx.org/logging/
         'threshold': 'BLOCK_ONLY_HIGH'}          # import logging; logging.basicConfig(level=logging.DEBUG)
    ]
}

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
    # Note: make sure to upgrade your Azure account from "free" to "pay as you go"
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
    } if keys.AZURE_API_KEY != 'MY_API_KEY' else None,

    'azure-sweden': {    # TODO ---- this is getting hacky and should be refactored asap    - split into clients.py file + make keys.py a dict
        'sync': AzureOpenAI(
            api_key=keys.AZURE_API_KEY_2,
            azure_endpoint=keys.AZURE_OPENAI_ENDPOINT_2,
            api_version='2024-02-01',   #'2022-12-01',
            http_client=http_client
        ),
        'async': AsyncAzureOpenAI(
            api_key=keys.AZURE_API_KEY_2,
            azure_endpoint=keys.AZURE_OPENAI_ENDPOINT_2,
            api_version='2024-02-01',   #'2022-12-01',
            http_client=http_async
        )
    } if keys.AZURE_API_KEY_2 != 'MY_API_KEY' else None,

    # https://docs.together.ai/reference/completions
    'together': {
        'sync': Together(
            api_key=keys.TOGETHER_API_KEY,
        ),
        'async': AsyncTogether(
            api_key=keys.TOGETHER_API_KEY,
        )
    } if keys.TOGETHER_API_KEY != 'MY_API_KEY' else None,

    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library
    # model="google/gemini-1.5-flash-001"
    'google': {
        'sync': OpenAI(
            base_url=f"https://{keys.GOOGLE_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{keys.GOOGLE_PROJECT}/locations/{keys.GOOGLE_LOCATION}/endpoints/openapi",
            api_key=GOOGLE_CREDENTIALS.token,
            http_client=http_client,
        ),
        'async': AsyncOpenAI(
            base_url=f"https://{keys.GOOGLE_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{keys.GOOGLE_PROJECT}/locations/{keys.GOOGLE_LOCATION}/endpoints/openapi",
            api_key=GOOGLE_CREDENTIALS.token,
            http_client=http_async
        )
    } if GOOGLE_CREDENTIALS is not None else None,
}

def refresh_google_creds():
    if GOOGLE_CREDENTIALS is None:
        return
    GOOGLE_CREDENTIALS.refresh(transport.requests.Request())
    CLIENTS['google']['sync'].api_key = GOOGLE_CREDENTIALS.token
    CLIENTS['google']['async'].api_key = GOOGLE_CREDENTIALS.token

refresh_google_creds()


# Model map for services like Azure that index by deployments instead of model names
# This is needed for async batching across multiple backends simultaneously
MODEL_MAP = {
    'azure': {
        'gpt-3.5-turbo-0613': 'gpt-35-0613',    #  gpt-3.5-turbo-0125
        'gpt-3.5-turbo-instruct-0914': 'gpt-35-instruct-0914'
    }
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
            if type(v) is not dict:
                v = v or 0    # replace None with 0
                usage[k] += v
    pprint(dict(usage))


def run_async(call_fn, inputs_ls, model_args, verbose=1, queries_per_batch=1,    # TODO: rename `queries_per_batch` to `prompts_per_query`
              backend='openai', concurrency=1000, is_chat=False):
    qpb = queries_per_batch
    n_inputs = len(inputs_ls)
    concurrency = min(concurrency, n_inputs)

    get_inputs = lambda i: inputs_ls[i : i+qpb] if qpb > 1 else inputs_ls[i]

    ####### TODO
    if backend == 'google':
        concurrency = min(300, n_inputs)
        refresh_google_creds()

    ######################

    async def f(n1, n2):
        n2 = min(n2, n_inputs)

        # Setup async tasks for a single backend
        if type(backend) is str:
            backend_idxs = defaultdict(lambda: backend)   # map from async_call index -> backend
            async_calls = [asyncio.create_task(
                call_fn(backend)(get_inputs(i), **model_args)) for i in np.arange(n1, n2, qpb)]

        # Multiple backends   -  dict mapping from backend to TPM, e.g. {'openai': 90, 'azure': 240}
        elif type(backend) is dict:
            total_tpm = sum(backend.values())

            async_range = np.arange(n1, n2, qpb)
            num_calls = len(async_range)

            i1, i2 = 0, 0
            async_calls  = [None] * num_calls
            backend_idxs = [None] * num_calls

            # Split batches according to relative TPM
            for backend_, tpm in backend.items():

                m_args = model_args.copy()
                if 'azure' in backend_:
                    m_args['model'] = MODEL_MAP['azure'][m_args['model']]

                bk_calls = np.ceil(num_calls * tpm / total_tpm).astype(int)

                i1 = i2
                i2 = min(i1 + bk_calls, num_calls)
                for i_ in range(i1, i2):
                    i = async_range[i_]
                    f = call_fn(backend_)(get_inputs(i), **m_args)

                    async_calls[i_] = asyncio.create_task(f)
                    backend_idxs[i_] = backend_
        else:
            raise TypeError(f'`backend` ({backend}) must be a string or dict from string to TPM counts')

        # Asynchronously collect this batch of data
        gather = asyncio.gather if verbose == 0 else tqdm_asyncio.gather
        completions = await gather(*async_calls)

        # Add `.backend` attribute to keep track of which API did which completion
        for i_, completion in enumerate(completions):
            completion.backend = backend_idxs[i_]
        return completions

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
retry = lambda x: retry_tenacity(
    wait=wait_random_exponential(max=100),
    stop=stop_after_attempt(200)
        )(timeout(20)(x))

# retry = retry_tenacity(wait=wait_random(), stop=stop_after_attempt(200))
# retry = lambda x: x          # Dummy retry func, useful for debugging
