from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def timing_event(event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        **fields,
    }
    print("TIMING " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")), flush=True)
