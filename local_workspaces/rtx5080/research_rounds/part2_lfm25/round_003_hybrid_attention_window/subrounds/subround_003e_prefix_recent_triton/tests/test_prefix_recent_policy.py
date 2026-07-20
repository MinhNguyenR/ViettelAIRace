from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

from policy import FULL_LAYERS, parse_window_map, prefix_ranges  # noqa: E402


def test_map_is_exact_and_positive() -> None:
    value = parse_window_map("2:4096,5:4096,8:4096,10:4096,12:4096,14:4096")
    assert set(value) == FULL_LAYERS
    assert all(item == 4096 for item in value.values())


def test_prefix_ranges_are_inclusive() -> None:
    assert prefix_ranges(3, 128) == {
        0: [(0, 127)],
        1: [(0, 127)],
        2: [(0, 127)],
    }


def test_zero_prefix_is_reserved_for_backend_control() -> None:
    try:
        prefix_ranges(1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero prefix must not create a synthetic range")


def test_invalid_map_rejected() -> None:
    try:
        parse_window_map("2:4096,5:4096")
    except ValueError:
        pass
    else:
        raise AssertionError("partial map must fail")
