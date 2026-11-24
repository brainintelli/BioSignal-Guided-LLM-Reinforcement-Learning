"""Deterministic physical state layouts for the new-world diagnostic."""


def canonical_batches(inputs, boundaries=(896, 1280, 1537), batch_size=4, pad_id=0):
    if (not boundaries or any(type(x) is not int or x < 1 for x in boundaries)
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Invalid canonical batch dimensions')
    if not inputs or any(not row or len(row) >= boundaries[-1] for row in inputs):
        raise ValueError('Empty or overflowing sequence; never truncate')
    buckets = {}
    for index, row in enumerate(inputs):
        # Keep a masked leading token even at a bucket boundary. Otherwise an
        # all-ones batch triggers SDPA's mask-elision path and changes numerics.
        width = next(n for n in boundaries if n > len(row))
        buckets.setdefault(width, []).append(index)
    for width, indices in sorted(buckets.items()):
        for start in range(0, len(indices), batch_size):
            real = indices[start:start + batch_size]
            physical = real + [real[-1]] * (batch_size - len(real))
            yield dict(indices=real, width=width,
                       input_ids=[[pad_id] * (width - len(inputs[i])) + list(inputs[i]) for i in physical],
                       attention_mask=[[0] * (width - len(inputs[i])) + [1] * len(inputs[i]) for i in physical])
