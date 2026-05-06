"""Đọc dữ liệu triều đại từ timeline.sqlite3 (read-only seed data)."""

from __future__ import annotations

from typing import Any

from db.connection import get_timeline_conn


def get_dynasties() -> list[dict[str, Any]]:
    conn = get_timeline_conn()
    try:
        rows = conn.execute(
            """
            SELECT d.*, k.id AS k_id, k."order" AS k_order,
                   k.name AS k_name, k.reign_start, k.reign_end,
                   k.description AS k_description, k.dynasty_id
            FROM core_dynasty d
            LEFT JOIN core_king k ON k.dynasty_id = d.id
            ORDER BY d."order", k."order"
            """
        ).fetchall()
    finally:
        conn.close()

    dynasties: dict[int, dict] = {}
    for r in rows:
        d_id = r["id"]
        if d_id not in dynasties:
            dynasties[d_id] = {
                "id": d_id,
                "order": r["order"],
                "name": r["name"],
                "start_year": r["start_year"],
                "end_year": r["end_year"],
                "description": r["description"],
                "color": r["color"] or "#8B6914",
                "kings": [],
            }
        if r["k_id"] is not None:
            dynasties[d_id]["kings"].append({
                "id": r["k_id"],
                "order": r["k_order"],
                "name": r["k_name"],
                "reign_start": r["reign_start"],
                "reign_end": r["reign_end"],
                "description": r["k_description"],
                "dynasty_id": r["dynasty_id"],
            })
    return list(dynasties.values())
