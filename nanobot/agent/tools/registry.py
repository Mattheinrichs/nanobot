"""Tool registry for dynamic tool management."""

import json
from collections import OrderedDict
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self, cache_results: bool = False, cache_max_size: int = 128):
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None
        self._cache_enabled = cache_results
        self._cache_max_size = cache_max_size
        self._result_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """Extract a normalized tool name from either OpenAI or flat schemas."""
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions with stable ordering for cache-friendly prompts.

        Built-in tools are sorted first as a stable prefix, then MCP tools are
        sorted and appended.  The result is cached until the next
        register/unregister call.
        """
        if self._cached_definitions is not None:
            return self._cached_definitions

        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []
        for schema in definitions:
            name = self._schema_name(schema)
            if name.startswith("mcp_"):
                mcp_tools.append(schema)
            else:
                builtins.append(schema)

        builtins.sort(key=self._schema_name)
        mcp_tools.sort(key=self._schema_name)
        self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    def prepare_call(
        self,
        name: str,
        params: dict[str, Any],
    ) -> tuple[Tool | None, dict[str, Any], str | None]:
        """Resolve, cast, and validate one tool call."""
        # Guard against invalid parameter types (e.g., list instead of dict)
        if not isinstance(params, dict) and name in ("write_file", "read_file"):
            return (
                None,
                params,
                (
                    f"Error: Tool '{name}' parameters must be a JSON object, got {type(params).__name__}. "
                    'Use named parameters: tool_name(param1="value1", param2="value2")'
                ),
            )

        tool = self._tools.get(name)
        if not tool:
            return (
                None,
                params,
                (f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"),
            )

        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return (
                tool,
                cast_params,
                (f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)),
            )
        return tool, cast_params, None

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"
        tool, params, error = self.prepare_call(name, params)
        if error:
            return error + _HINT

        try:
            assert tool is not None  # guarded by prepare_call()
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    async def execute_cached(self, name: str, tool: Tool, params: dict[str, Any]) -> Any:
        """Execute a tool, returning a cached result for read-only tools when caching is enabled.

        Only caches results for tools where ``tool.read_only`` is ``True``.
        Error results are never cached.  Uses LRU eviction once ``cache_max_size``
        is reached.  Falls back to direct execution if the params are not
        JSON-serialisable.
        """
        if not self._cache_enabled or not tool.read_only:
            return await tool.execute(**params)

        try:
            cache_key = (name, json.dumps(params, sort_keys=True, default=str))
        except Exception:
            return await tool.execute(**params)

        if cache_key in self._result_cache:
            logger.debug("Tool cache hit: {}", name)
            self._cache_hits += 1
            self._result_cache.move_to_end(cache_key)
            return self._result_cache[cache_key]

        self._cache_misses += 1
        result = await tool.execute(**params)

        # Don't cache errors — let the agent retry with different params.
        if not (isinstance(result, str) and result.startswith("Error")):
            if len(self._result_cache) >= self._cache_max_size:
                evicted = self._result_cache.popitem(last=False)
                logger.debug("Tool cache evicted LRU entry: {}", evicted[0][0])
            self._result_cache[cache_key] = result

        return result

    @property
    def cache_stats(self) -> dict[str, int | float]:
        """Return hit/miss counters for the result cache.

        Returns a dict with keys ``hits``, ``misses``, ``eligible`` (hits + misses),
        and ``hit_rate`` (0.0–1.0; 0.0 when no eligible calls have been made).
        """
        eligible = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "eligible": eligible,
            "hit_rate": self._cache_hits / eligible if eligible > 0 else 0.0,
        }

    def clear_result_cache(self) -> None:
        """Clear the tool result cache and log a hit-rate summary (e.g. at session reset)."""
        eligible = self._cache_hits + self._cache_misses
        if self._cache_enabled and eligible > 0:
            logger.info(
                "Tool cache stats: {}/{} hits ({:.0%})",
                self._cache_hits,
                eligible,
                self._cache_hits / eligible,
            )
        self._result_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
