# Batch-prompt

This is a lightweight wrapper for querying LLMs with batches of prompts. Supports OpenAI's Completion and ChatCompletion APIs. 

- Prompt batching with ChatCompletions API, [which does not support multiple prompts in a single API call](https://community.openai.com/t/batching-with-chatcompletion-not-possible-like-it-was-in-completion/81647). Efficiently execute many async API calls with retry and exponential backoff logic.
- Automatically generate raw prompts to query given a list of prompt templates, and a list of keywords. E.g. with the 2 prompts `['I like {food}', 'I hate {food}']` and the 2 keyword dicts `[{'food': 'pizza'}, {'food': 'apples'}]`, generate 4 prompt text strings: `['I like pizza', 'I like apples', I hate pizza', 'I hate apples']`.
- Unpack results and parse into a standard format, where each item (1) refers to a single LLM output ("choice"), and (2) contains relevant meta-data like the specific prompt and keyword dict used. Results dict is portable - it can be pickled and re-loaded in a runtime which only has `pickle` and `openai` installed.
- Split final API calls into smaller batches of prompts (`num_batches` argument), since API limits the number of tokens per LLM query.


Used for [In-Context Learning Dynamics with Random Binary Sequences](https://arxiv.org/abs/2310.17639), which involved querying GPT models with many batches of prompts.


##### Other similar libraries

Similar to [parallel-parrot](https://github.com/novex-ai/parallel-parrot), but more general and lightweight. [LangChain](https://github.com/langchain-ai/langchain) has some analogous functionality with generating prompts based on templates + kwarg dicts, but the interface and codebase is more elaborate since batch prompting isn't the main goal.



## Example use


TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 

TODO - give simple example and examine the results object




## Installation and setup

1. Update the variables in `batch_prompt/keys.py` with their OpenAI API and organization keys.
2. Append the `batch_prompt` directory to your system path, e.g.:

```lang=python
import sys
sys.path.append('/path/to/batch-prompt/batch_prompt')
```





## Future plans

- Could be generalized beyond OpenAI LLMs if other API clients use a similar enough interface
- Easier installation: `setup.py` for direct installation; add to pypi if other people find this package useful
