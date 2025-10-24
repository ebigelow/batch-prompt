from time import time
from pprint import pprint

from batch_prompt.utils import retry, CLIENTS, print_call_summary, run_async, simplify_completion, GOOGLE_SAFETY


DEFAULT_MODEL_ARGS = {
    'model': 'gpt-3.5-turbo-0613',  #'gpt-3.5-turbo-0125',
    'n': 1,
    'max_tokens': 5,
    'temperature': 1.0,
}


def chat_async_backoff(backend):
    @retry
    async def f(messages, *args, **kwargs):
        # Note: extra_body is a dict that gets merged with request json  
        #     https://github.com/openai/openai-python/blob/45315a/src/openai/_base_client.py#L453
        extra_body = (GOOGLE_SAFETY if backend == 'google' else None)
        client = CLIENTS[backend]['async']
        return await client.chat.completions.create(messages=messages, extra_body=extra_body, *args, **kwargs)
    return f


def flatten(ls):
    return [i for subl in ls for i in subl]

def listify_messages(messages, messages_args=None):
    if len(messages) > 0 and type(messages[0]) is list: 
        return flatten([listify_messages(msgs, messages_args) for msgs in messages])
    if messages_args and type(messages_args[0]) is list:
        return flatten([listify_messages(messages, msg_args) for msg_args in messages_args])

    messages_args = messages_args or [{} for _ in messages]
    return [(messages, messages_args)]


def get_chat_completions(messages, messages_args=None, model_args=None, verbose=2, 
                         backend='openai', zip_model_args=False, **kwargs):
    # Note: chat completions does not work in jupyter due to async
    messages_ls, messages_args_ls = zip(*listify_messages(messages, messages_args))
    formatted_msgs = [[{'role': m['role'], 
                        'content': m['content'].format(**kwargs) if kwargs else m['content']}
                       for m, kwargs in zip(msgs, msg_args)]
                      for msgs, msg_args in zip(messages_ls, messages_args_ls)]

    # Model Args
    m_args = DEFAULT_MODEL_ARGS.copy()
    if type(model_args) is dict:
        m_args.update(model_args)
    elif type(model_args) is list:
        m_args = [(m_args | m) for m in model_args]

    if verbose > 1:
        print('='*80)
        print(f'Calling {backend} Chat API . . .')
        t1 = time()
    if verbose > 2:
        print('------- Messages -------')
        pprint(formatted_msgs)

    completions = run_async(chat_async_backoff, formatted_msgs, m_args, verbose, 
                            backend=backend, is_chat=True, **kwargs)

    # Return list of formatted dictionaries
    results = [
        {
            'choice': c.dict(),
            'completion': simplify_completion(completion),
            'messages': formatted_msgs[idx],
            'messages_raw': messages_ls[idx],
            'messages_args': messages_args_ls[idx],
            'model_args': model_args[idx] if (type(model_args) is list) else model_args,
        }
        for idx, completion in enumerate(completions)
        for c in completion.choices
    ]
    if verbose > 1:
        print_call_summary(t1, len(results), completions)

    return results

