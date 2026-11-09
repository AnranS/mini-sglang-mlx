"""
Radix 前缀缓存，对应原版 minisgl/kvcache/radix_cache.py（SGLang 的 RadixAttention）。

一棵基数树（压缩前缀树）：每个节点存一段 token 序列 key 和对应的 KV 槽位 value。
从根到某节点的路径拼起来，就是一个"曾经算过的前缀"。

    root
     ├── [你好, 我是]  ──┬── [小明]          请求 A: 你好 我是 小明
     │                  └── [小红, 。]      请求 B: 你好 我是 小红 。
     └── [Hello]

新请求到来时沿树向下匹配最长前缀，命中部分的 KV 直接复用，不用重新 prefill。

几个要点：
  * 引用计数 ref_count：有请求正在使用某条路径时，路径上所有节点 ref_count > 0，不能被淘汰。
  * 淘汰 evict：按 LRU（timestamp）从叶子开始淘汰 ref_count == 0 的节点。
  * page_size > 1 时，只能按整页匹配/插入（子节点用"第一页的 token 元组"做 key）。

整个结构都在 CPU 上（numpy），和原版一样 —— 树操作本来就是 CPU 逻辑。
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from minisgl_mlx.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KeyFn = Callable[[np.ndarray], Any]


def fast_compare_key(a: np.ndarray, b: np.ndarray) -> int:
    """返回 a 和 b 的公共前缀长度。原版用 C++ 实现，这里用 numpy 向量化。"""
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) > 0 else n


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KeyFn, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()
        self._key: np.ndarray
        self._value: np.ndarray
        self._length: int = 0

    def set_key_value(self, key: np.ndarray, value: np.ndarray) -> None:
        assert len(key) == len(value)
        self._key, self._value, self._length = key, value, len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def key(self) -> np.ndarray:
        return self._key

    @property
    def value(self) -> np.ndarray:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: np.ndarray) -> int:
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """把当前节点在 pos 处一分为二：返回新的父节点（前半段），自己变成后半段。"""
        assert 0 < pos < self.length
        parent = self.parent
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)
        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> np.ndarray:
        values: List[np.ndarray] = []
        node = self.node
        while not node.is_root():
            values.append(node.value)
            node = node.parent
        if not values:
            return np.empty(0, dtype=np.int32)
        values.reverse()
        return np.concatenate(values)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, page_size: int = 1) -> None:
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # 根节点永远受保护

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: np.ndarray) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: np.ndarray, indices: np.ndarray) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # 有一段新的后缀，挂成新的叶子
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:].copy(), indices[prefix_len:].copy())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> np.ndarray:
        if size == 0:
            return np.empty(0, dtype=np.int32)
        assert size <= self.evictable_size, f"Cannot evict {size}, only {self.evictable_size} evictable"

        leaves = self._collect_evictable_leaves()
        heapq.heapify(leaves)  # 按 timestamp 排序：最久没用的先淘汰（LRU）
        evicted: List[np.ndarray] = []
        evicted_size = 0
        while evicted_size < size:
            assert leaves, f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leaves)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node.key)]
            # 父节点可能因此变成叶子，也加入候选（根节点 ref_count=1，永远不会被淘汰）
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leaves, parent)
        return np.concatenate(evicted)

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        evictable = protected = 0
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            if node.ref_count == 0:
                evictable += node.length
            else:
                protected += node.length
            stack.extend(node.children.values())
        assert (evictable, protected) == (self.evictable_size, self.protected_size), (
            f"Radix integrity check failed: counted ({evictable}, {protected}) "
            f"!= recorded ({self.evictable_size}, {self.protected_size})"
        )

    def pretty_print(self, tokenizer=None, max_tokens: int = 8) -> str:
        """把树打印出来，方便学习时观察（tutorials/ch05_radix_cache.py 会用到）。"""
        lines: List[str] = ["root"]

        def fmt(node: RadixTreeNode) -> str:
            ids = node.key[:max_tokens].tolist()
            text = repr(tokenizer.decode(ids)) if tokenizer is not None else str(ids)
            more = "..." if node.length > max_tokens else ""
            return f"{text}{more} (len={node.length}, ref={node.ref_count})"

        def walk(node: RadixTreeNode, prefix: str) -> None:
            children = sorted(node.children.values(), key=lambda n: n.uuid)
            for i, child in enumerate(children):
                last = i == len(children) - 1
                lines.append(prefix + ("└── " if last else "├── ") + fmt(child))
                walk(child, prefix + ("    " if last else "│   "))

        walk(self.root_node, "")
        return "\n".join(lines)

    def _collect_evictable_leaves(self) -> List[RadixTreeNode]:
        stack, leaves = [self.root_node], []
        while stack:
            node = stack.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def _tree_walk(self, input_ids: np.ndarray) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()
        while prefix_len < total:
            if total - prefix_len < self.page_size:
                break  # 剩下的不足一页，无法匹配
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            # 至少匹配了一页（key 就是第一页），所以 match_len >= page_size
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:  # 只匹配了节点的一部分：分裂节点
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic  # 更新 LRU 时间戳
        return node, prefix_len


def _get_key_fn(page_size: int) -> KeyFn:
    if page_size == 1:
        return lambda x: int(x[0])
    return lambda x: tuple(x[:page_size].tolist())
