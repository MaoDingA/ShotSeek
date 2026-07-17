"""Exact conversions between milliseconds and CFR frame indices."""

from __future__ import annotations

from fractions import Fraction


def parse_ratio(value: str) -> Fraction:
    if not value or value == "0/0":
        raise ValueError("missing rational value")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid rational value: {value}") from exc
    if result <= 0:
        raise ValueError("rational value must be positive")
    return result


def frame_to_ms(frame: int, fps: Fraction) -> int:
    """Round an exact frame boundary to its nearest integer millisecond."""
    if frame < 0 or fps <= 0:
        raise ValueError("frame must be non-negative and fps positive")
    value = Fraction(frame * 1000, 1) / fps
    return (value.numerator * 2 + value.denominator) // (2 * value.denominator)


def ms_to_floor_frame(milliseconds: int, fps: Fraction) -> int:
    if milliseconds < 0 or fps <= 0:
        raise ValueError("milliseconds must be non-negative and fps positive")
    value = Fraction(milliseconds, 1000) * fps
    return value.numerator // value.denominator


def ms_to_ceil_frame(milliseconds: int, fps: Fraction) -> int:
    if milliseconds < 0 or fps <= 0:
        raise ValueError("milliseconds must be non-negative and fps positive")
    value = Fraction(milliseconds, 1000) * fps
    return -(-value.numerator // value.denominator)


def pts_to_frame(pts: int, time_base: Fraction, fps: Fraction) -> int:
    if pts < 0 or time_base <= 0 or fps <= 0:
        raise ValueError("pts must be non-negative and ratios positive")
    value = pts * time_base * fps
    nearest = (value.numerator * 2 + value.denominator) // (2 * value.denominator)
    if abs(value - nearest) > Fraction(1, 1000):
        raise ValueError(f"PTS {pts} does not map to a CFR frame: {value}")
    return nearest
