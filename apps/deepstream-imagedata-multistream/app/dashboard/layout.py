from __future__ import annotations

import math


def wall_layout(camera_count: int, preferred_columns: int = 6) -> tuple[int, int]:
    """Return rows/columns that expose every configured camera tile."""
    count = max(0, int(camera_count))
    if count == 0:
        return 1, 1
    columns = max(1, min(int(preferred_columns), count))
    return max(1, math.ceil(count / columns)), columns
