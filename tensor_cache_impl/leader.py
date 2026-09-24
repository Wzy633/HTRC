"""Deprecated compatibility import; canonical implementation is ``tensors``."""

from tensors.leader import CacheLeader, LEADER

f3cLeader = CacheLeader
__all__ = ["CacheLeader", "f3cLeader", "LEADER"]
