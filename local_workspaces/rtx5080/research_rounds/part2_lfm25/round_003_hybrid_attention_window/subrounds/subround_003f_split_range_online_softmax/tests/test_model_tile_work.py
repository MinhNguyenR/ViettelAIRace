from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_tile_work import model_step, tile_sets  # noqa: E402


def test_prefix_does_not_add_future_keys() -> None:
    recent, _native, split = tile_sets(0, 4000, 0, 128)
    assert split == recent


def test_late_prefix_adds_only_prefix_tiles() -> None:
    recent, native, split = tile_sets(20_784, 4_208, 0, 128)
    assert len(split - recent) == 4
    assert len(native) > len(split)


def test_native_mm_scans_full_sequence_for_each_query_block() -> None:
    result = model_step(20_784, 4_208, 128)
    assert result["native_mm"] > result["ideal_split"]
    assert result["ideal_split"] > result["recent"]


def test_prefix_inside_recent_window_adds_no_tile_work() -> None:
    result = model_step(0, 4_000, 128)
    assert result["ideal_split"] == result["recent"]
