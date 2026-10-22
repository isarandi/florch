"""Tests for the worker-safe barecat reader."""

import os.path as osp
import tempfile

import barecat
import pytest

from florch.barecat_reader import BarecatReader


@pytest.fixture
def archive():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = osp.join(tmpdir, 'test.barecat')
        with barecat.Barecat(path, readonly=False) as bc:
            bc['imgs/a.npy'] = b'a'
            bc['imgs/sub/b.npy'] = b'b'
            bc['top.npy'] = b'top'
        yield path


def test_len_counts_files(archive):
    """num_files is a property; calling it made every len() raise TypeError."""
    reader = BarecatReader(archive)
    try:
        assert len(reader) == 3
    finally:
        reader.close()


def test_keys_yields_only_files(archive):
    """iter_all_paths() is dirs UNION files, so it also yielded '' and every dir."""
    reader = BarecatReader(archive)
    try:
        keys = sorted(reader.keys())
        assert keys == ['imgs/a.npy', 'imgs/sub/b.npy', 'top.npy']
        assert '' not in keys
        assert 'imgs' not in keys
    finally:
        reader.close()


def test_len_agrees_with_keys(archive):
    """The pair drifted apart before; keep them pinned to each other."""
    reader = BarecatReader(archive)
    try:
        assert len(reader) == len(list(reader.keys()))
    finally:
        reader.close()


def test_every_key_is_readable(archive):
    """Building a dataset from keys() used to hit directories and fail."""
    reader = BarecatReader(archive)
    try:
        for key in reader.keys():
            assert isinstance(reader[key], bytes)
    finally:
        reader.close()
