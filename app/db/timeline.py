"""Đọc dữ liệu triều đại — PostgreSQL hoặc SQLite (read-only seed data)."""

from __future__ import annotations

from typing import Any

from app.db.connection import _use_postgres, get_pool, get_timeline_conn


async def get_dynasties() -> list[dict[str, Any]]:
    if _use_postgres():
        return await _get_dynasties_pg()
    return _get_dynasties_sqlite()


async def _get_dynasties_pg() -> list[dict[str, Any]]:
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d."order", d.name, d.start_year, d.end_year,
                   d.description, d.color,
                   k.id AS k_id, k."order" AS k_order,
                   k.name AS k_name, k.reign_start, k.reign_end,
                   k.description AS k_description, k.dynasty_id
            FROM core_dynasty d
            LEFT JOIN core_king k ON k.dynasty_id = d.id
            ORDER BY d."order", k."order"
            """
        )
    return _build_dynasties(rows, pg=True)


def _get_dynasties_sqlite() -> list[dict[str, Any]]:
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
    return _build_dynasties(rows, pg=False)


def _build_dynasties(rows, *, pg: bool) -> list[dict[str, Any]]:
    def val(row, key):
        return row[key] if pg else row[key]

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
