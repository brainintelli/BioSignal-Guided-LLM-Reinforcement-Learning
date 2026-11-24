"""Lossless token-boundary validation for the separate speed experiment."""


def split_prefix(inputs, prefix, max_tokens):
    if not inputs or not prefix or max_tokens < 1:
        raise ValueError('Nonempty inputs, prefix and a positive limit are required')
    result = []
    for ids in inputs:
        if len(ids) > max_tokens:
            raise ValueError('Context overflow; legal choices must never be truncated')
        if len(ids) <= len(prefix) or ids[:len(prefix)] != prefix:
            raise ValueError('Template prefix differs or has no continuation')
        result.append(ids[len(prefix):])
    return result
