"""Features temporales: hora del día, día de semana, densidad de mensajes."""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def extract(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Devuelve un DataFrame con features temporales."""
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    out = pd.DataFrame(index=df.index)
    out["hour_of_day"] = ts.dt.hour.fillna(-1).astype(int)
    out["day_of_week"] = ts.dt.dayofweek.fillna(-1).astype(int)
    out["is_night"] = ((out["hour_of_day"] >= 22) | (out["hour_of_day"] < 6)).astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["month"] = ts.dt.month.fillna(-1).astype(int)

    # Densidad: #mensajes por usuario (proxy de cadencia).
    counts = df.groupby("user_id").size().rename("msg_per_user")
    out = out.join(counts, on="user_id").fillna({"msg_per_user": 0})
    out["msg_per_user"] = out["msg_per_user"].astype(int)
    return out


__all__ = ["extract"]
