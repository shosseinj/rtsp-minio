from __future__ import annotations

from enum import Enum


class FullscreenState(str, Enum):
    DISABLED = "DISABLED"
    SWITCHING = "SWITCHING"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"


VALID_TRANSITIONS = {
    FullscreenState.DISABLED: {FullscreenState.SWITCHING},
    FullscreenState.SWITCHING: {
        FullscreenState.ACTIVE, FullscreenState.DRAINING
    },
    FullscreenState.ACTIVE: {
        FullscreenState.SWITCHING, FullscreenState.DRAINING
    },
    FullscreenState.DRAINING: {FullscreenState.DISABLED},
}


def validate_transition(
    current: FullscreenState, target: FullscreenState
) -> None:
    if target not in VALID_TRANSITIONS[current]:
        raise ValueError(f"invalid fullscreen transition: {current}->{target}")
