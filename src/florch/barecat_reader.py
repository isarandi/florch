"""Worker-safe barecat reader for multi-process data loading."""

import os
from typing import Any, Optional

import barecat


class BarecatReader:
    """Worker-safe barecat reader that auto-reinitializes per process.

    When using multi-process data loading (fork or spawn), each worker process
    needs its own file handles and database connections. This class automatically
    detects when it's in a new process and reinitializes the underlying Barecat instance.

    Example:
        from barecat import DecodedView

        images = BarecatReader('images.barecat')
        annos = BarecatReader('annos.barecat')

        def load_item(key, *, rng, group_rng):
            d = DecodedView(annos)[key]
            image = DecodedView(images)[d['impath']]
            return dict(image=image, label=d['label'])
    """

    def __init__(self, path: str, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self._bc: Optional[barecat.Barecat] = None
        self._pid: Optional[int] = None

    def _ensure_open(self) -> barecat.Barecat:
        """Open barecat if not open or if we're in a new process."""
        pid = os.getpid()
        if self._bc is None or self._pid != pid:
            if self._bc is not None:
                try:
                    self._bc.close()
                except Exception:
                    pass
            self._bc = barecat.Barecat(self.path, readonly=True, threadsafe=True, **self.kwargs)
            self._pid = pid
        return self._bc

    def __getitem__(self, key: str) -> Any:
        return self._ensure_open()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._ensure_open()

    def get(self, key: str, default: Any = None) -> Any:
        bc = self._ensure_open()
        try:
            return bc[key]
        except KeyError:
            return default

    def keys(self):
        # Files only: iter_all_paths() is dirs UNION files, so it also yields the
        # root '' and every directory, and reader[dirpath] then fails.
        return self._ensure_open().index.iter_all_filepaths()

    def __len__(self) -> int:
        # num_files is a property; calling it raised TypeError on every len().
        return self._ensure_open().index.num_files

    def close(self):
        if self._bc is not None:
            self._bc.close()
            self._bc = None
            self._pid = None

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"BarecatReader({self.path!r})"

    def __getstate__(self):
        return {'path': self.path, 'kwargs': self.kwargs}

    def __setstate__(self, state):
        self.path = state['path']
        self.kwargs = state['kwargs']
        self._bc = None
        self._pid = None
