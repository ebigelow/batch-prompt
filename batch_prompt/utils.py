from collections import defaultdict
from time import time
from pprint import pprint

import openai
from batch_prompt import keys

if keys.API_KEY == "MY_API_KEY":
    raise ImportError('Add your OpenAI API key to keys.py')

openai.organization = keys.ORGANIZATION
openai.api_key = keys.API_KEY


COST_1k_TOK = {
    'gpt-3.5-turbo-instruct': [.0015, .002],
    'gpt-3.5-turbo': [.0005, .0015],
    'gpt-4': [.05, .05],
    'davinci-002': [.002, .002],
    'babbage-002': [.0004, .0004],
}

def print_call_summary(t1, n_res, completions):
    """Helper method to print summary stats and query useage for a batch of calls."""
    t2 = time()
    print('~'*15, 'Done', '~'*15)
    print(f'Time: {t2 - t1 :.2f}s')
    print(f'Number of results: {n_res}')

    # Aggregate usage tokens across batches
    usage = defaultdict(lambda: 0)
    for completion in completions:
        for k, v in completion['usage'].items():
            usage[k] += v
    pprint(dict(usage))
