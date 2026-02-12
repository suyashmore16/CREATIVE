from pathlib import Path
from datetime import datetime
import json

def write_session_metadata(
    subject_id: str,
    srate: float,
    n_channels: int,
    channels: list[str],
    notes: str = "",
):
    meta = {
        "subject_id": subject_id,
        "datetime": datetime.now().isoformat(),
        "sampling_rate_hz": srate,
        "n_channels": n_channels,
        "channels": channels,
        "conditions": ["baseline", "stress", "recovery"],
        "notes": notes,
    }

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{subject_id}_metadata.json"

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[META] Wrote session metadata → {path}")
