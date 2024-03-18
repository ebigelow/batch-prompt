# Batch-prompt

This is a lightweight wrapper for querying LLMs with batches of prompts. Supports OpenAI's Completion and ChatCompletion APIs. 

- Prompt batching with ChatCompletions API, [which does not support multiple prompts in a single API call](https://community.openai.com/t/batching-with-chatcompletion-not-possible-like-it-was-in-completion/81647). Execute many async API calls with retry and exponential backoff logic.
- Automatically generate raw prompts to query given a list of prompt templates, and a list of keywords. E.g. with the 2 prompts `['I like {food}', 'I hate {food}']` and the 2 keyword dicts `[{'food': 'pizza'}, {'food': 'apples'}]`, generate 4 prompt text strings: `['I like pizza', 'I like apples', I hate pizza', 'I hate apples']`.
- Unpack results and parse into a standard format, where each item (1) refers to a single LLM output ("choice"), and (2) contains relevant meta-data like the specific prompt and keyword dict used. Results dict is portable - can be pickled and re-loaded in a runtime which only has `openai` installed, `batch_prompt` not required.
- Split final API calls into smaller batches of prompts (`queries_per_batch` argument), since API limits the number of tokens per LLM query. Note: Completions API is not async, since OpenAI has tighter limits on requests per min. and the API supports batching multiple inputs in a request.


Used for [In-Context Learning Dynamics with Random Binary Sequences](https://arxiv.org/abs/2310.17639), which involved querying GPT models with many batches of prompts.


##### Other similar libraries

Similar to [parallel-parrot](https://github.com/novex-ai/parallel-parrot), but more general and lightweight. [LangChain](https://github.com/langchain-ai/langchain) has some analogous functionality with generating prompts based on templates + kwarg dicts, but the interface and codebase is more elaborate since batch prompting isn't the main goal.





## Installation and setup

1. Update the variables in `batch_prompt/keys.py` with their OpenAI API and organization keys.
2. Append the `batch_prompt` directory to your system path, e.g.:

```python
import sys
sys.path.append('/path/to/batch-prompt')
```




## Example use



### Minimal example

```python
p = """Q: Are the following coin flips from a random coin flip, or non-random coin flip? {sequence}

A: The flips are from a"""

res = batch_prompt.completions(
    p, prompt_args={'sequence': '[Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails]'}, 
    model_args={'max_tokens': 1, 'logprobs': 5})
```

where `res` is structured as:

```
[{'prompt': 'Q: Are the following coin flips from a random coin flip, or non-random coin flip? [Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails]\n\nA: The flips are from a',
  'choice': <OpenAI Choice>,
  'completion': <OpenAI Completion>,
  'prompt_raw': 'Q: Are the following coin flips from a random coin flip, or non-random coin flip? {sequence}\n\nA: The flips are from a',
  'prompt_args': {'sequence': '[Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails, Heads, Tails]'},
  'model_args': {'max_tokens': 1, 'logprobs': 5}}]
```


Each result corresponds to a single OpenAI Choice object, as well as a pointer to the Completion Object.
A Completion object has a list `completion.choices`, where one Choice corresponds to a single model output. See [API docs](https://platform.openai.com/docs/api-reference/completions/object#completions/object-choices) for full structure.

This format is useful by:
1. Flattening results, so that one result matches one LLM output. This helps with batching lots of queries and different prompts, and not worrying about nesting model outputs depending on how prompts were batched.
2. Recording the exact prompt and keyword arguments used. This means that with lots of queries, or with merging batches of data queried at different times, it's easy to parse back the exact prompt used as well as prompt keywords, which are useful for indexing (e.g. when {sequence} in the prompt was formatted as 'Heads, Tails, Heads')
3. Simple dict can be easily pickled and loaded without having `batch-prompt`, and are json-serializable if the OpenAI objects are converted to json.




### Batching

```python
prompts = ['I like {food}', 'I hate {food}']

res = batch_prompt.completions(
    prompts, prompt_args=[{'food': 'pizza'}, {'food': 'apples'}], 
    model_args={'max_tokens': 10, 'n': 2})


>>> res
[
 {'prompt': 'I like pizza',
  'choice': <OpenAIObject at 0x111c3e590> JSON: {
    "text": ".\n\nThat's surprising.\n\nRight? Is that not",
    "index": 0,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I like {food}',
  'prompt_args': {'food': 'pizza'},
  'model_args': {'max_tokens': 10, 'n': 2}},
 {'prompt': 'I like pizza',
  'choice': <OpenAIObject at 0x10e7288b0> JSON: {
    "text": "\n\nPizza is a very popular and delicious food",
    "index": 1,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I like {food}',
  'prompt_args': {'food': 'pizza'},
  'model_args': {'max_tokens': 10, 'n': 2}},
 {'prompt': 'I like apples',
  'choice': <OpenAIObject at 0x111c12950> JSON: {
    "text": "\n\nApples are a delicious fruit that come in",
    "index": 2,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I like {food}',
  'prompt_args': {'food': 'apples'},
  'model_args': {'max_tokens': 10, 'n': 2}},
 {'prompt': 'I like apples',
  'choice': <OpenAIObject at 0x111c12ea0> JSON: {
    "text": ".\"\n\nApples are a nutritious and delicious fruit",
    "index": 3,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I like {food}',
  'prompt_args': {'food': 'apples'},
  'model_args': {'max_tokens': 10, 'n': 2}},
 {'prompt': 'I hate pizza',
  'choice': <OpenAIObject at 0x111c124a0> JSON: {
    "text": " too.\n\nOh my God, get out of here",
    "index": 4,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I hate {food}',
  'prompt_args': {'food': 'pizza'},
  'model_args': {'max_tokens': 10, 'n': 2}},
 {'prompt': 'I hate pizza',
  'choice': <OpenAIObject at 0x111c12860> JSON: {
    "text": ".\n\nGo back and try again.\n\nI love pizza",
    "index": 5,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I hate {food}',
  'prompt_args': {'food': 'pizza'},
  'model_args': {'max_tokens': 10, 'n': 2}},
 {'prompt': 'I hate apples',
  'choice': <OpenAIObject at 0x111c12e00> JSON: {
    "text": ".\n\nI don't hate apples. apples, but",
    "index": 6,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I hate {food}',
  'prompt_args': {'food': 'apples'},
  'model_args': {'max_tokens': 10, 'n': 2}},
 {'prompt': 'I hate apples',
  'choice': <OpenAIObject at 0x111c2f770> JSON: {
    "text": ". I can't stand the texture and the taste",
    "index": 7,
    "logprobs": null,
    "finish_reason": "length"
  },
  'completion': <OpenAIObject text_completion id=cmpl-8qOYTC... >,
  'prompt_raw': 'I hate {food}',
  'prompt_args': {'food': 'apples'},
  'model_args': {'max_tokens': 10, 'n': 2}
 }
]
```


### Chat API

```python
p1 = 'I like {food}'
p2 = 'I hate {food}'
f1 = 'pizza'
f2 = 'apples'


msgs = [[{'role': 'user', 'content': p1}], [{'role': 'user', 'content': p2}]]
res = batch_prompt.chat_completions(
   msgs, messages_args=[[{'food': f1}], [{'food': f2}]], 
   model_args={'max_tokens': 10, 'n': 2}, verbose=3)
```

```
>>> from pprint import pprint
>>> for r in res:
>>>    pprint({k: v for k,v in r.items() if k != 'completion'})

{'completion_tokens': 80, 'prompt_tokens': 40, 'total_tokens': 120}
{'choice': {'finish_reason': 'length',
            'index': 0,
            'logprobs': None,
            'message': {'content': "That's great! Pizza is a popular and "
                                   'delicious',
                        'role': 'assistant'}},
 'messages': [{'content': 'I like pizza', 'role': 'user'}],
 'messages_args': [{'food': 'pizza'}],
 'messages_raw': [{'content': 'I like {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
{'choice': {'finish_reason': 'length',
            'index': 1,
            'logprobs': None,
            'message': {'content': "That's great! Pizza is a popular food "
                                   'enjoyed',
                        'role': 'assistant'}},
 'messages': [{'content': 'I like pizza', 'role': 'user'}],
 'messages_args': [{'food': 'pizza'}],
 'messages_raw': [{'content': 'I like {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
{'choice': {'finish_reason': 'length',
            'index': 0,
            'logprobs': None,
            'message': {'content': "That's great to hear! Apples are a",
                        'role': 'assistant'}},
 'messages': [{'content': 'I like apples', 'role': 'user'}],
 'messages_args': [{'food': 'apples'}],
 'messages_raw': [{'content': 'I like {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
{'choice': {'finish_reason': 'length',
            'index': 1,
            'logprobs': None,
            'message': {'content': "That's great! Apples are a delicious and",
                        'role': 'assistant'}},
 'messages': [{'content': 'I like apples', 'role': 'user'}],
 'messages_args': [{'food': 'apples'}],
 'messages_raw': [{'content': 'I like {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
{'choice': {'finish_reason': 'length',
            'index': 0,
            'logprobs': None,
            'message': {'content': "I'm sorry to hear that. Pizza is a",
                        'role': 'assistant'}},
 'messages': [{'content': 'I hate pizza', 'role': 'user'}],
 'messages_args': [{'food': 'pizza'}],
 'messages_raw': [{'content': 'I hate {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
{'choice': {'finish_reason': 'length',
            'index': 1,
            'logprobs': None,
            'message': {'content': "That's okay, everyone has different "
                                   'preferences when it',
                        'role': 'assistant'}},
 'messages': [{'content': 'I hate pizza', 'role': 'user'}],
 'messages_args': [{'food': 'pizza'}],
 'messages_raw': [{'content': 'I hate {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
 {'choice': {'finish_reason': 'length',
            'index': 0,
            'logprobs': None,
            'message': {'content': "I'm sorry to hear that. Apples are",
                        'role': 'assistant'}},
 'messages': [{'content': 'I hate apples', 'role': 'user'}],
 'messages_args': [{'food': 'apples'}],
 'messages_raw': [{'content': 'I hate {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
{'choice': {'finish_reason': 'length',
            'index': 1,
            'logprobs': None,
            'message': {'content': 'I understand that not everyone enjoys the '
                                   'taste of apples',
                        'role': 'assistant'}},
 'messages': [{'content': 'I hate apples', 'role': 'user'}],
 'messages_args': [{'food': 'apples'}],
 'messages_raw': [{'content': 'I hate {food}', 'role': 'user'}],
 'model_args': {'max_tokens': 10, 'n': 2}}
]
```






## Future plans

- [x] Simple test examples for chat + completions
- [x] Batch chat completions with arbitrary messages list
- [x] Async non-chat completions
- [x] Update to use `openai >= 1.0.0` -- client object
- [ ] Generalize beyond OpenAI LLMs: integrate with [pyllms](https://github.com/kagisearch/pyllms/tree/main)
- [ ] Easier installation: `setup.py` for direct installation; add to pypi if other people find this package useful

- [ ] (maybe?) Subprocess instead of import to enable async calls from jupyter
