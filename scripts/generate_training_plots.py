"""Generate training evidence plots from backend JSONL metrics.

Run from the repository root:
  python scripts/generate_training_plots.py

The script intentionally reads only persisted backend metrics. It skips missing
JSONL files gracefully and never fabricates rewards or losses.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Iterable


METRIC_FILES = {
    "q_learning": Path("artifacts/q_learning/metrics.jsonl"),
    "neural_policy": Path("artifacts/neural_policy/metrics.jsonl"),
    "grpo": Path("artifacts/grpo/metrics.jsonl"),
    "multi_agent_grpo": Path("artifacts/multi_agent_grpo/metrics.jsonl"),
}

ASSET_DIR = Path("assets")
REWARD_PNG = ASSET_DIR / "reward_curve.png"
LOSS_PNG = ASSET_DIR / "loss_curve.png"
SUMMARY_JSON = ASSET_DIR / "training_summary.json"

COLORS = {
    "q_learning": (52, 211, 153),
    "neural_policy": (96, 165, 250),
    "grpo": (251, 191, 36),
    "multi_agent_grpo": (244, 114, 182),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def metric_rows() -> dict[str, list[dict]]:
    return {mode: read_jsonl(path) for mode, path in METRIC_FILES.items()}


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw.append(0)
        start = y * row_bytes
        raw.extend(pixels[start : start + row_bytes])
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        pixels[offset : offset + 3] = bytes(color)


def draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for ox in (0, 1):
            for oy in (0, 1):
                set_pixel(pixels, width, height, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(max(0, top), min(height, bottom + 1)):
        for x in range(max(0, left), min(width, right + 1)):
            set_pixel(pixels, width, height, x, y, color)


def collect_series(rows_by_mode: dict[str, list[dict]], metric: str) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for mode, rows in rows_by_mode.items():
        points: list[tuple[float, float]] = []
        for index, row in enumerate(rows, start=1):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            x_value = row.get("update_step") if metric == "policy_loss" else row.get("episode")
            if not isinstance(x_value, (int, float)):
                x_value = index
            points.append((float(x_value), float(value)))
        if points:
            series[mode] = points
    return series


def plot_series(path: Path, series: dict[str, list[tuple[float, float]]]) -> None:
    width, height = 960, 540
    bg = (11, 15, 25)
    grid = (39, 49, 68)
    axis = (148, 163, 184)
    pixels = bytearray(bg * width * height)
    left, right, top, bottom = 70, width - 35, 35, height - 65

    for i in range(6):
        y = top + round((bottom - top) * i / 5)
        draw_line(pixels, width, height, (left, y), (right, y), grid)
    for i in range(6):
        x = left + round((right - left) * i / 5)
        draw_line(pixels, width, height, (x, top), (x, bottom), grid)
    draw_line(pixels, width, height, (left, top), (left, bottom), axis)
    draw_line(pixels, width, height, (left, bottom), (right, bottom), axis)

    all_points = [point for points in series.values() for point in points]
    if not all_points:
        write_png(path, width, height, pixels)
        return

    min_x = min(x for x, _ in all_points)
    max_x = max(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_y = max(y for _, y in all_points)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    y_pad = max((max_y - min_y) * 0.08, 0.01)
    min_y -= y_pad
    max_y += y_pad

    def scale(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        sx = left + round((x - min_x) / (max_x - min_x) * (right - left))
        sy = bottom - round((y - min_y) / (max_y - min_y) * (bottom - top))
        return sx, sy

    legend_x = left
    for mode, points in series.items():
        color = COLORS.get(mode, (255, 255, 255))
        draw_rect(pixels, width, height, legend_x, 14, legend_x + 26, 24, color)
        legend_x += 45
        scaled = [scale(point) for point in points]
        for start, end in zip(scaled, scaled[1:]):
            draw_line(pixels, width, height, start, end, color)
        for x, y in scaled:
            draw_rect(pixels, width, height, x - 2, y - 2, x + 2, y + 2, color)

    write_png(path, width, height, pixels)


def latest_number(rows: Iterable[dict], key: str) -> float | None:
    for row in reversed(list(rows)):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def build_summary(rows_by_mode: dict[str, list[dict]]) -> dict:
    used_files = [path.as_posix() for mode, path in METRIC_FILES.items() if rows_by_mode.get(mode)]
    all_rows = [row for rows in rows_by_mode.values() for row in rows]
    latest_by_mode: dict[str, dict] = {}
    for mode, rows in rows_by_mode.items():
        if not rows:
            continue
        latest = rows[-1]
        rewards = [float(row["total_reward"]) for row in rows if isinstance(row.get("total_reward"), (int, float))]
        latest_by_mode[mode] = {
            "latest_reward": latest.get("total_reward"),
            "best_reward": max(rewards) if rewards else None,
            "latest_task_score": latest.get("task_score"),
            "latest_final_status": latest.get("final_status"),
            "latest_policy_loss": latest_number(rows, "policy_loss"),
            "latest_checkpoint_path": latest.get("checkpoint_path"),
            "latest_metrics_path": latest.get("metrics_path"),
        }

    rewards = [float(row["total_reward"]) for row in all_rows if isinstance(row.get("total_reward"), (int, float))]
    latest = all_rows[-1] if all_rows else {}
    return {
        "generated_from": used_files,
        "latest_reward": latest.get("total_reward"),
        "best_reward": max(rewards) if rewards else None,
        "latest_task_score": latest.get("task_score"),
        "latest_final_status": latest.get("final_status"),
        "latest_policy_loss": latest_number(all_rows, "policy_loss"),
        "modes": latest_by_mode,
    }


def main() -> int:
    rows_by_mode = metric_rows()
    ASSET_DIR.mkdir(exist_ok=True)
    plot_series(REWARD_PNG, collect_series(rows_by_mode, "total_reward"))
    plot_series(LOSS_PNG, collect_series(rows_by_mode, "policy_loss"))
    SUMMARY_JSON.write_text(json.dumps(build_summary(rows_by_mode), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {REWARD_PNG.as_posix()}")
    print(f"Wrote {LOSS_PNG.as_posix()}")
    print(f"Wrote {SUMMARY_JSON.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
