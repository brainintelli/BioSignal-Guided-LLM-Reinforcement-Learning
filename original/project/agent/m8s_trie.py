"""A constrained semantic sequence distribution with explicit end-of-choice."""


def choice_tree(sequences,eos):
    if not sequences or any(not row for row in sequences):raise ValueError('Empty semantic choices')
    if any(eos in row for row in sequences):raise ValueError('EOS inside a semantic label')
    if len({tuple(row) for row in sequences})!=len(sequences):raise ValueError('Duplicate semantic token sequence')
    children={}
    for row in sequences:
        seq=list(row)+[eos]
        for i,token in enumerate(seq):children.setdefault(tuple(seq[:i]),set()).add(token)
    nodes={prefix:sorted(tokens) for prefix,tokens in sorted(children.items(),key=lambda x:(len(x[0]),x[0])) if len(tokens)>1}
    paths=[]
    for row in sequences:
        path=[]
        for i,token in enumerate(list(row)+[eos]):
            prefix=tuple(row[:i])
            if prefix in nodes:path.append((prefix,nodes[prefix].index(token)))
        paths.append(path)
    return nodes,paths
