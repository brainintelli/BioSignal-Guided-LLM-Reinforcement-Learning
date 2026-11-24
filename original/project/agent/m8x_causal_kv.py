"""Explicitly registered empty-thinking target-template normalization."""


def normalize_target(verb_ids,target_ids,boundary,limit):
    if boundary<1 or verb_ids[:boundary]!=target_ids[:boundary] or boundary>=len(verb_ids):
        raise ValueError('Target does not share the declared assistant boundary')
    result=list(verb_ids)+list(target_ids[boundary:])
    if len(result)>limit:raise ValueError('Normalized target exceeds context limit')
    return result
