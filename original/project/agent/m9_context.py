"""Lossless M9 overflow buckets; historical <=1536 layouts remain identical."""


def context_layout(max_tokens):
    if type(max_tokens) is not int or max_tokens not in (1536, 4096):
        raise ValueError('Unregistered M9 context capacity')
    return (896, 1280, 1537) if max_tokens == 1536 else (896, 1280, 1537, 2049, 3073, 4097)


def tokenize_observation(tokenizer, messages, max_tokens):
    context_layout(max_tokens)
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Target-trie continuations share the context. Reserve space for the verb,
    # target and batching padding; the legacy path keeps its original behavior.
    state_limit = max_tokens if max_tokens == 1536 else max_tokens - 64
    if len(ids) > state_limit:
        raise ValueError(f'Observation {len(ids)} exceeds {state_limit}; no silent truncation')
    return ids
