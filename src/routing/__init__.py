"""Local-versus-hosted routing: policy, fallback, circuit breaker, and answer cache."""

from src.routing.client import DEGRADED_EN, DEGRADED_FA, RouteDecision, RoutingClient

__all__ = ["DEGRADED_EN", "DEGRADED_FA", "RouteDecision", "RoutingClient"]
