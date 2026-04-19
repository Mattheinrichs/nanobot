"""Tests for ToolRegistry tool-result caching (Stage 2B)."""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ReadOnlyTool(Tool):
    """Minimal read-only tool for testing."""

    def __init__(self, name: str, return_value: Any = "result"):
        self._name = name
        self._return_value = return_value
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test read-only tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"q": {"type": "string"}}}

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._return_value


class _MutableTool(Tool):
    """Minimal mutable (non-read-only) tool for testing."""

    def __init__(self, name: str = "write_tool"):
        self._name = name
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test mutable tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def read_only(self) -> bool:
        return False

    async def execute(self, **kwargs: Any) -> Any:
        self.call_count += 1
        return "written"


# ---------------------------------------------------------------------------
# Cache disabled (default behaviour unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_disabled_by_default_always_calls_tool():
    registry = ToolRegistry()  # cache_results=False by default
    tool = _ReadOnlyTool("search")
    registry.register(tool)

    await registry.execute_cached("search", tool, {"q": "hello"})
    await registry.execute_cached("search", tool, {"q": "hello"})

    assert tool.call_count == 2


# ---------------------------------------------------------------------------
# Cache enabled — read-only tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_returns_same_result_without_re_executing():
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search", return_value="cached_value")
    registry.register(tool)

    result1 = await registry.execute_cached("search", tool, {"q": "hello"})
    result2 = await registry.execute_cached("search", tool, {"q": "hello"})

    assert result1 == "cached_value"
    assert result2 == "cached_value"
    assert tool.call_count == 1  # only executed once


@pytest.mark.asyncio
async def test_different_params_are_cached_separately():
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search")
    tool.call_count = 0

    await registry.execute_cached("search", tool, {"q": "hello"})
    await registry.execute_cached("search", tool, {"q": "world"})
    await registry.execute_cached("search", tool, {"q": "hello"})  # cache hit

    assert tool.call_count == 2


@pytest.mark.asyncio
async def test_params_order_irrelevant_for_cache_key():
    """json.dumps with sort_keys=True means param order doesn't affect the key."""
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search")

    await registry.execute_cached("search", tool, {"b": 2, "a": 1})
    await registry.execute_cached("search", tool, {"a": 1, "b": 2})

    assert tool.call_count == 1


# ---------------------------------------------------------------------------
# Cache enabled — mutable tools must never be cached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mutable_tool_is_never_cached():
    registry = ToolRegistry(cache_results=True)
    tool = _MutableTool()
    registry.register(tool)

    await registry.execute_cached("write_tool", tool, {})
    await registry.execute_cached("write_tool", tool, {})

    assert tool.call_count == 2


# ---------------------------------------------------------------------------
# Error results are not cached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_result_is_not_cached():
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search", return_value="Error: something went wrong")
    registry.register(tool)

    await registry.execute_cached("search", tool, {"q": "fail"})
    await registry.execute_cached("search", tool, {"q": "fail"})

    assert tool.call_count == 2  # retried because error was not cached


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lru_eviction_removes_oldest_entry():
    registry = ToolRegistry(cache_results=True, cache_max_size=2)
    tool = _ReadOnlyTool("search")

    await registry.execute_cached("search", tool, {"q": "a"})  # cache: [a]
    await registry.execute_cached("search", tool, {"q": "b"})  # cache: [a, b]
    await registry.execute_cached("search", tool, {"q": "c"})  # evicts a; cache: [b, c]

    assert tool.call_count == 3

    # "a" was evicted — miss; evicts b; cache: [c, a]
    await registry.execute_cached("search", tool, {"q": "a"})
    assert tool.call_count == 4

    # "c" is still cached (MRU slot; "b" was the evicted LRU)
    await registry.execute_cached("search", tool, {"q": "c"})
    assert tool.call_count == 4

    # "b" was evicted — should re-execute
    await registry.execute_cached("search", tool, {"q": "b"})
    assert tool.call_count == 5


@pytest.mark.asyncio
async def test_lru_access_refreshes_eviction_order():
    """Accessing an entry should move it to most-recently-used, protecting it from eviction."""
    registry = ToolRegistry(cache_results=True, cache_max_size=2)
    tool = _ReadOnlyTool("search")

    await registry.execute_cached("search", tool, {"q": "a"})  # cache: [a]
    await registry.execute_cached("search", tool, {"q": "b"})  # cache: [a, b]
    # Access "a" again to make it most-recently-used
    await registry.execute_cached("search", tool, {"q": "a"})  # cache: [b, a]
    # Now adding "c" should evict "b", not "a"
    await registry.execute_cached("search", tool, {"q": "c"})  # evicts b; cache: [a, c]

    assert tool.call_count == 3  # a×1, b×1, c×1

    # "a" and "c" should still be cached
    await registry.execute_cached("search", tool, {"q": "a"})
    await registry.execute_cached("search", tool, {"q": "c"})
    assert tool.call_count == 3

    # "b" should have been evicted
    await registry.execute_cached("search", tool, {"q": "b"})
    assert tool.call_count == 4


# ---------------------------------------------------------------------------
# clear_result_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_result_cache_forces_re_execution():
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search")

    await registry.execute_cached("search", tool, {"q": "hello"})
    assert tool.call_count == 1

    registry.clear_result_cache()

    await registry.execute_cached("search", tool, {"q": "hello"})
    assert tool.call_count == 2


# ---------------------------------------------------------------------------
# cache_stats
# ---------------------------------------------------------------------------


def test_cache_stats_initial_all_zeros():
    registry = ToolRegistry(cache_results=True)
    stats = registry.cache_stats
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["eligible"] == 0
    assert stats["hit_rate"] == 0.0


@pytest.mark.asyncio
async def test_cache_stats_counts_hits_and_misses():
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search")

    # First call: miss
    await registry.execute_cached("search", tool, {"q": "a"})
    # Second call same params: hit
    await registry.execute_cached("search", tool, {"q": "a"})
    # Third call different params: miss
    await registry.execute_cached("search", tool, {"q": "b"})

    stats = registry.cache_stats
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["eligible"] == 3


@pytest.mark.asyncio
async def test_cache_stats_hit_rate():
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search")

    await registry.execute_cached("search", tool, {"q": "x"})  # miss
    await registry.execute_cached("search", tool, {"q": "x"})  # hit
    await registry.execute_cached("search", tool, {"q": "x"})  # hit

    stats = registry.cache_stats
    assert stats["hit_rate"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_cache_stats_not_incremented_when_cache_disabled():
    registry = ToolRegistry(cache_results=False)
    tool = _ReadOnlyTool("search")

    await registry.execute_cached("search", tool, {"q": "a"})
    await registry.execute_cached("search", tool, {"q": "a"})

    stats = registry.cache_stats
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["eligible"] == 0


@pytest.mark.asyncio
async def test_cache_stats_not_incremented_for_mutable_tools():
    registry = ToolRegistry(cache_results=True)
    tool = _MutableTool()

    await registry.execute_cached("write_tool", tool, {})
    await registry.execute_cached("write_tool", tool, {})

    stats = registry.cache_stats
    assert stats["hits"] == 0
    assert stats["misses"] == 0


@pytest.mark.asyncio
async def test_clear_result_cache_resets_counters():
    registry = ToolRegistry(cache_results=True)
    tool = _ReadOnlyTool("search")

    await registry.execute_cached("search", tool, {"q": "a"})  # miss
    await registry.execute_cached("search", tool, {"q": "a"})  # hit

    assert registry.cache_stats["hits"] == 1
    assert registry.cache_stats["misses"] == 1

    registry.clear_result_cache()

    stats = registry.cache_stats
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["eligible"] == 0
    assert stats["hit_rate"] == 0.0
