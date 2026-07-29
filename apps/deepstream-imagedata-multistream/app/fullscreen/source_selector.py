from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRoute:
    source_id: str
    generation: int
    mux_index: int


class SourceRouteTable:
    def __init__(self, routes: list[SourceRoute]) -> None:
        self._routes = {route.source_id: route for route in routes}

    def resolve(self, source_id: str, generation: int) -> SourceRoute:
        route = self._routes.get(source_id)
        if route is None:
            raise KeyError(source_id)
        if route.generation != generation:
            raise ValueError("stale_generation")
        return route
