from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


MODULE_PATH = Path(__file__).parents[1] / "overlay" / "lfm2_all_shortconv.py"


def load_overlay() -> types.ModuleType:
    torch_stub = types.ModuleType("torch")
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    try:
        spec = importlib.util.spec_from_file_location("r002aa_test_overlay", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_torch is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous_torch


class CacheConfig:
    block_size = 16


class Request:
    def __init__(self, *, computed: int, tokens: int) -> None:
        self.num_computed_tokens = computed
        self.num_tokens = tokens


class Block:
    def __init__(self, block_id: int, *, is_null: bool = False) -> None:
        self.block_id = block_id
        self.is_null = is_null


class BlockPool:
    def __init__(self) -> None:
        self.next_id = 10_000

    def get_new_blocks(self, count: int) -> list[Block]:
        blocks = [Block(self.next_id + offset) for offset in range(count)]
        self.next_id += count
        return blocks


class FakeScalar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def item(self) -> bool:
        return self.value


class Vector:
    def __init__(self, values: list[int | bool]) -> None:
        self.values = values

    def _binary(self, other: object, operation: object) -> Vector:
        right = other.values if isinstance(other, Vector) else [other] * len(self.values)
        return Vector([operation(left, value) for left, value in zip(self.values, right)])

    def __getitem__(self, key: int | slice) -> int | bool | Vector:
        value = self.values[key]
        return Vector(value) if isinstance(key, slice) else value

    def __sub__(self, other: object) -> Vector:
        return self._binary(other, lambda left, right: left - right)

    def __mod__(self, other: object) -> Vector:
        return self._binary(other, lambda left, right: left % right)

    def __gt__(self, other: object) -> Vector:
        return self._binary(other, lambda left, right: left > right)

    def __lt__(self, other: object) -> Vector:
        return self._binary(other, lambda left, right: left < right)

    def __and__(self, other: object) -> Vector:
        return self._binary(other, lambda left, right: bool(left and right))


class FakeTorch:
    @staticmethod
    def diff(values: Vector) -> Vector:
        return Vector(
            [right - left for left, right in zip(values.values, values.values[1:])]
        )

    @staticmethod
    def any(values: Vector) -> FakeScalar:
        return FakeScalar(any(values.values))


class SparseControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.overlay = load_overlay()
        self.overlay._SPARSE_MODE_FILE = ""
        self.overlay._SPARSE_RUNTIME_ENABLED = True

    def test_scheduler_bypasses_only_final_crossing_split(self) -> None:
        class Scheduler:
            def __init__(self) -> None:
                self.cache_config = CacheConfig()

            def schedule(self) -> str:
                return "scheduled"

            def _mamba_block_aligned_split(
                self,
                request: Request,
                num_new_tokens: int,
                num_new_local_computed_tokens: int = 0,
                num_external_computed_tokens: int = 0,
            ) -> int:
                return 777

        module = types.SimpleNamespace(Scheduler=Scheduler)
        self.overlay._patch_scheduler(module)
        scheduler = Scheduler()
        request = Request(computed=4000, tokens=8199)

        self.assertEqual(scheduler._mamba_block_aligned_split(request, 4199), 4199)
        self.assertEqual(scheduler._mamba_block_aligned_split(request, 4192), 777)
        cold_request = Request(computed=0, tokens=4001)
        self.assertEqual(
            scheduler._mamba_block_aligned_split(cold_request, 4001), 777
        )
        self.overlay._SPARSE_RUNTIME_ENABLED = False
        self.assertEqual(scheduler._mamba_block_aligned_split(request, 4199), 777)

    def test_manager_reserves_and_installs_one_boundary_block(self) -> None:
        null = Block(-1, is_null=True)
        final = Block(9000)

        class MambaManager:
            def __init__(self) -> None:
                self.mamba_cache_mode = "align"
                self.block_size = 16
                self._allocated_block_reqs: set[str] = set()
                self.req_to_blocks: dict[str, list[Block]] = {}
                self.block_pool = BlockPool()

            def get_num_blocks_to_allocate(
                self,
                request_id: str,
                num_tokens: int,
                new_computed_blocks: object,
                total_computed_tokens: int,
                num_tokens_main_model: int,
                apply_admission_cap: bool = False,
            ) -> int:
                return 2

            def allocate_new_blocks(
                self,
                request_id: str,
                num_tokens: int,
                num_tokens_main_model: int,
            ) -> list[Block]:
                self._allocated_block_reqs.add(request_id)
                self.req_to_blocks[request_id] = [null] * 512 + [final]
                return [final]

        module = types.SimpleNamespace(MambaManager=MambaManager)
        self.overlay._patch_mamba_manager(module)
        manager = MambaManager()

        count = manager.get_num_blocks_to_allocate(
            "req", 4199, [], 4000, 8199
        )
        self.assertEqual(count, 3)
        cold_count = manager.get_num_blocks_to_allocate(
            "cold", 4001, [], 0, 4001
        )
        self.assertEqual(cold_count, 2)
        allocated = manager.allocate_new_blocks("req", 4199, 8199)
        boundary = manager.req_to_blocks["req"][511]
        self.assertFalse(boundary.is_null)
        self.assertEqual(manager.req_to_blocks["req"][512], final)
        self.assertEqual(allocated, [boundary, final])

    def test_non_sparse_all_mode_never_reads_align_only_state(self) -> None:
        class MambaManager:
            mamba_cache_mode = "all"

            def allocate_new_blocks(
                self,
                request_id: str,
                num_tokens: int,
                num_tokens_main_model: int,
            ) -> list[str]:
                return ["native"]

            def get_num_blocks_to_allocate(self, *args: object, **kwargs: object) -> int:
                return 1

        module = types.SimpleNamespace(MambaManager=MambaManager)
        self.overlay._patch_mamba_manager(module)
        self.overlay._SPARSE_RUNTIME_ENABLED = False
        self.assertEqual(
            MambaManager().allocate_new_blocks("req", 8, 8), ["native"]
        )

    def test_cold_metadata_fails_closed_to_native_path(self) -> None:
        metadata = types.SimpleNamespace(num_prefills=1)

        class MetadataBuilder:
            def __init__(self) -> None:
                cache_config = types.SimpleNamespace(mamba_cache_mode="align")
                self.vllm_config = types.SimpleNamespace(cache_config=cache_config)
                self.kv_cache_spec = types.SimpleNamespace(block_size=16)

            def _compute_common_metadata(self, common: object) -> object:
                return metadata

        module = types.SimpleNamespace(
            BaseMambaAttentionMetadataBuilder=MetadataBuilder
        )
        self.overlay._patch_mamba_metadata(module)
        self.overlay.torch = FakeTorch
        builder = MetadataBuilder()
        cold = types.SimpleNamespace(
            seq_lens_cpu_upper_bound=Vector([4001]),
            query_start_loc_cpu=Vector([0, 4001]),
            num_reqs=1,
        )

        result = builder._compute_common_metadata(cold)
        self.assertIs(result, metadata)
        self.assertFalse(hasattr(result, "r002aa_sparse_dual"))

        missing_cpu_lengths = types.SimpleNamespace(
            seq_lens_cpu_upper_bound=None,
            num_reqs=1,
        )
        self.assertIs(
            builder._compute_common_metadata(missing_cpu_lengths), metadata
        )

    def test_mode_file_switches_runtime_without_repatching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mode_file = Path(temporary) / "mode.txt"
            self.overlay._SPARSE_MODE_FILE = str(mode_file)
            self.overlay._SPARSE_RUNTIME_ENABLED = False
            mode_file.write_text("1\n")
            self.overlay._refresh_sparse_mode()
            self.assertTrue(self.overlay._sparse_enabled())
            mode_file.write_text("0\n")
            self.overlay._refresh_sparse_mode()
            self.assertFalse(self.overlay._sparse_enabled())
            mode_file.write_text("invalid\n")
            with self.assertRaises(RuntimeError):
                self.overlay._refresh_sparse_mode()


if __name__ == "__main__":
    unittest.main()
