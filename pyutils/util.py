from matplotlib.pylab import f


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

def scinot_str(num, standard_notation_ok=True, standard_notation_threshold=0.001):
    """
    Given a number produce a string with that number in scientific notation
    """

    if standard_notation_ok and abs(num) >= standard_notation_threshold:
        return f"{num:.3f}"
    else:
        s = f"{num:2.1e}"
        s = r"$" + s.replace("e", r" \times 10^{") + r"}$"
        return s