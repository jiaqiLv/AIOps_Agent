"""Lazy graph proxy for langgraph dev compatibility.

Defers graph construction (and LLM initialization) until first use,
so modules can be imported without valid API credentials.
"""

import threading
from typing import Any


class LazyGraph:
    """Proxy that defers CompiledStateGraph construction until first access.

    Usage:
        graph = LazyGraph(build_my_agent)
        # graph acts like a CompiledStateGraph but only builds on first use.
    """

    def __init__(self, builder):
        self._builder = builder
        self._graph = None
        self._lock = threading.Lock()

    def _ensure_built(self):
        if self._graph is None:
            with self._lock:
                if self._graph is None:
                    self._graph = self._builder()

    # ---- proxied methods / attributes ----

    def invoke(self, *args, **kwargs) -> Any:
        self._ensure_built()
        return self._graph.invoke(*args, **kwargs)

    def get_graph(self, *args, **kwargs) -> Any:
        self._ensure_built()
        return self._graph.get_graph(*args, **kwargs)

    def stream(self, *args, **kwargs) -> Any:
        self._ensure_built()
        return self._graph.stream(*args, **kwargs)

    def astream(self, *args, **kwargs) -> Any:
        self._ensure_built()
        return self._graph.astream(*args, **kwargs)

    @property
    def nodes(self):
        self._ensure_built()
        return self._graph.nodes

    @property
    def edges(self):
        self._ensure_built()
        return self._graph.edges

    @property
    def builder(self):
        self._ensure_built()
        return self._graph.builder

    def __getattr__(self, name: str) -> Any:
        self._ensure_built()
        return getattr(self._graph, name)

    def __repr__(self) -> str:
        if self._graph is None:
            return f"<LazyGraph (not built yet)>"
        return repr(self._graph)
