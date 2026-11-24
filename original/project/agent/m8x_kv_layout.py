"""Token identity checks for branchable, unpadded Qwen continuations."""


def checked_suffix(ids, prefix, limit):
    if not prefix or len(ids)>limit or ids[:len(prefix)]!=list(prefix):
        raise ValueError('Context overflow or non-identical shared prefix')
    suffix=ids[len(prefix):]
    # Qwen single-token decode mutates convolution state and uses a different
    # recurrent kernel. This layout deliberately uses multi-token continuations.
    if len(suffix)<2:
        raise ValueError('At least two continuation tokens are required')
    return suffix
