from __future__ import annotations

import json
from pathlib import Path


class DurableOutbox:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict, snapshot: bytes | None = None) -> Path:
        folder = self.root / event["event_id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "event.json").write_text(
            json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if snapshot:
            (folder / "person.jpg").write_bytes(snapshot)
        (folder / "state.json").write_text(
            json.dumps({"status": event["status"]}), encoding="utf-8"
        )
        return folder

    def pending(self) -> list[Path]:
        return sorted(self.root.glob("*/event.json"))

    def mark(self, folder: Path, **values: object) -> None:
        path = folder / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update(values)
        path.write_text(json.dumps(state), encoding="utf-8")
