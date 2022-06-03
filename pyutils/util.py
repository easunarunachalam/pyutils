def flatten(d, parent_key='', sep='_'):
    """
    Flatten nested dictionaries
    """
    
    items = []
    for k, v in d.items():
        new_key = str(parent_key) + str(sep) + str(k) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
