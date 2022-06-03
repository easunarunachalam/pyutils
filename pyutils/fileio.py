import numpy as np
import os
from pathlib import Path
import pickle


def contains_any_targets(test, targets):
    return np.any([(target in test) for target in targets])

def list_files(folder=".", pattern=None, exclude_in_names=["calibration",]):
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            path = Path(os.path.join(root, filename))
            if (pattern is None) or (path.match(pattern)):
                if not contains_any_targets(str(path), exclude_in_names):
                    yield path

def pickle_save(picklepath, variable_to_save):
    """
    Convenience function to pickle variable
    """
    with open(picklepath, "wb") as fp:
        pickle.dump(variable_to_save, fp)

def pickle_load(picklepath):
    """
    Convenience function to load pickled variable
    """
    with open(picklepath, "rb") as fp:
        return pickle.load(fp)
