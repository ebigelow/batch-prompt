# Batch-prompt

This is a lightweight wrapper for querying LLMs with batches of prompts. Supports OpenAI's Completion and ChatCompletion APIs. 

- Prompt batching with ChatCompletions API, [which does not support multiple prompts in a single API call](https://community.openai.com/t/batching-with-chatcompletion-not-possible-like-it-was-in-completion/81647). Execute many async API calls with retry and exponential backoff logic.
- Automatically generate raw prompts to query given a list of prompt templates, and a list of keywords. E.g. with the 2 prompts `['I like {food}', 'I hate {food}']` and the 2 keyword dicts `[{'food': 'pizza'}, {'food': 'apples'}]`, generate 4 prompt text strings: `['I like pizza', 'I like apples', I hate pizza', 'I hate apples']`.
- Unpack results and parse into a standard format, where each item (1) refers to a single LLM output ("choice"), and (2) contains relevant meta-data like the specific prompt and keyword dict used. Results dict is portable - can be pickled and re-loaded in a runtime which only has `openai` installed, `batch_prompt` not required.
- Split final API calls into smaller batches of prompts (`num_batches` argument), since API limits the number of tokens per LLM query. Note: Completions API is not async, since OpenAI has tighter limits on requests per min. and the API supports batching multiple inputs in a request.


Used for [In-Context Learning Dynamics with Random Binary Sequences](https://arxiv.org/abs/2310.17639), which involved querying GPT models with many batches of prompts.


##### Other similar libraries

Similar to [parallel-parrot](https://github.com/novex-ai/parallel-parrot), but more general and lightweight. [LangChain](https://github.com/langchain-ai/langchain) has some analogous functionality with generating prompts based on templates + kwarg dicts, but the interface and codebase is more elaborate since batch prompting isn't the main goal.



## Example use


```lang=python
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



### Chat API

TODO





## Installation and setup

1. Update the variables in `batch_prompt/keys.py` with their OpenAI API and organization keys.
2. Append the `batch_prompt` directory to your system path, e.g.:

```lang=python
import sys
sys.path.append('/path/to/batch-prompt/batch_prompt')
```





## Future plans

- [ ] Simple test examples for chat + completions
- [ ] Batch chat completions with arbitrary messages list
- [ ] Generalize beyond OpenAI LLMs: integrate with [pyllms](https://github.com/kagisearch/pyllms/tree/main)
- [ ] Easier installation: `setup.py` for direct installation; add to pypi if other people find this package useful
