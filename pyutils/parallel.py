from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

def ProgressParallelGen(total, n_jobs=6):
    """
    Return
    """
    class ProgressParallel(Parallel):
        def __call__(self, *args, **kwargs):
            with tqdm(total=total) as self._pbar:
                return Parallel.__call__(self, *args, **kwargs)

        def print_progress(self):
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

    return ProgressParallel(n_jobs=n_jobs)
