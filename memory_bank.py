from typing import Self
import numpy as np
import torch
import torch.nn as nn
from collections import deque, abc
from collections import abc


class ReplacementPolicy(abc):
    def __call__(self: Self, candidate: torch.Tensor, comparator: nn.Module):
        raise NotImplementedError


class RandomReplace(ReplacementPolicy):
    """
    Replaces the comparator with the given probability.
    If no probability is given, the default probability is 0.5.
    """

    def __init__(self: Self, retention_prob: float = 0.5):
        if not 0 <= retention_prob <= 1:
            raise ValueError("Threshold must be a valid probability between 0 and 1.")
        self.threshold = retention_prob

    def __call__(self: Self, candidate: torch.Tensor, comparator: nn.Module) -> bool:
        return np.random.rand() < self.threshold


class AlwaysReplace(ReplacementPolicy):
    """
    Always replaces the comparator.
    """

    def __call__(self: Self, candidate: torch.Tensor, comparator: nn.Module) -> bool:
        """
        Returns whether the candidate should be allowed to replace the comparator.
        """
        return True


class DeltaReplace(ReplacementPolicy):
    """
    Replaces the comparator if the delta between the candidate and the comparator
    is greater than the delta_threshold.
    """

    def __init__(self: Self, delta_threshold: float, delta_fn: nn.Module):
        self.delta_threshold = delta_threshold
        self.delta_fn = delta_fn

    def __call__(self: Self, candidate: torch.Tensor, comparator: nn.Module) -> bool:
        return self.delta_fn(candidate, comparator) > self.delta_threshold


SELECTIVITY = 0.5

REPLACEMENT_CONFIG = {
    "random": RandomReplace(retention_prob=SELECTIVITY),
    "always_replace": AlwaysReplace(),
    "delta_replace": DeltaReplace(
        delta_threshold=SELECTIVITY, delta_fn=nn.CosineSimilarity
    ),
}

MEMORY_CONFIG = {
    "capacity": 16,
    "replacement_policy": REPLACEMENT_CONFIG["delta_replace"],
}


class MemoryBank(object):
    """
    Stores a collection of previously-observed frame embeddings.
    """

    def __init__(
        self: Self,
        embedding_dim: int,
        capacity: int = MEMORY_CONFIG["capacity"],
        replacement_policy: ReplacementPolicy = MEMORY_CONFIG["replacement_policy"],
    ):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.replacement_policy = replacement_policy
        self.memory = deque([], maxlen=capacity)

    def __len__(self: Self):
        return len(self.memory)

    def push(self: Self, candidate: torch.Tensor):
        """
        Attempts to push the candidate into the memory bank. If the memory bank is not full,
        the candidate is appended to the memory. If the memory bank is full, the candidate is
        compared against the newest element in the memory bank. If the replacement policy
        allows the candidate to replace the last element, the candidate is appended to the
        memory bank and the oldest element is removed.
        """
        if len(self) < self.capacity:
            self.memory.append(candidate)
            return True
        elif self.replacement_policy(candidate, self.memory[-1]):
            self.memory.popleft()
            self.memory.append(candidate)
            return True
        return False

    def clear(self: Self):
        self.memory.clear()

    def to_tensor(self: Self):
        """
        Returns a tensor containing the memory bank's contents.
        """
        return torch.stack(list(self.memory))
