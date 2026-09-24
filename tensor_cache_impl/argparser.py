"""Deprecated CLI compatibility wrapper."""

from tensors.argparser import parse_args


def parse_f3c_args():
    return parse_args()


__all__ = ["parse_args", "parse_f3c_args"]
