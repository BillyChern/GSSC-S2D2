"""The evaluation scratch-space guard, and the near-miss that motivated it.

Stage 1 of an evaluation writes one ``.label`` per frame into a temporary directory. On a
container that directory usually lands on the small overlay backing ``/``, so a full
SemanticKITTI val run (4071 frames, ~15.4 GiB) can fill the ROOT filesystem. The failure is
not a failed command: the host wedges, and this project has lost a machine to it repeatedly.

The regression these tests pin is subtler than "check that it fits". The first version of the
guard did exactly that -- and it PASSED the real configuration that prompted it, because
``/tmp`` offered 21.4 GiB against a 15.4 GiB requirement. Fitting with 6 GiB to spare still
leaves the root filesystem near-full for everything else on the box. So the guard reserves
headroom, and ``test_rejects_the_real_near_miss`` replays the measured numbers to keep it
that way.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from gssc.inference.evaluate import (
    _BYTES_PER_FRAME,
    _MIN_RESERVE,
    _assert_scratch_space,
    _count_frames,
)

#: The 4071-frame SemanticKITTI val sequence, the run every headline number is scored on.
VAL_FRAMES = 4071


def _usage(total_gib: float, free_gib: float) -> object:
    used = int((total_gib - free_gib) * 2**30)
    return shutil._ntuple_diskusage(int(total_gib * 2**30), used, int(free_gib * 2**30))


def _on_device(dev: int):
    return mock.patch("os.stat", side_effect=lambda p: mock.Mock(st_dev=dev))


def test_rejects_the_real_near_miss() -> None:
    """THE regression: /tmp as it actually was -- 50 GiB total, 21.4 GiB free, on root.

    A bare fits/does-not-fit test passes this, which is why the guard reserves headroom.
    """
    with mock.patch("shutil.disk_usage", return_value=_usage(50, 21.4)), _on_device(1):
        with pytest.raises(RuntimeError, match="ROOT filesystem"):
            _assert_scratch_space(Path("/tmp"), VAL_FRAMES)


def test_passes_when_the_volume_has_real_room() -> None:
    """Negative control: a guard that blocks a run that fits would be worse than none."""
    with mock.patch("shutil.disk_usage", return_value=_usage(6100, 696)), _on_device(2):
        _assert_scratch_space(Path("/data/scratch"), VAL_FRAMES)


def test_zero_frames_does_not_silently_demand_zero_bytes(caplog: pytest.LogCaptureFixture) -> None:
    """A miscounted frame total must disable the check LOUDLY, not pass it quietly.

    ``_count_frames`` returns 0 when the dataset layout is not the one it expects. Multiplied
    out that is a zero-byte requirement, i.e. a guard that always passes -- the bug that was
    actually shipped in the first draft and found only because the count was asserted.
    """
    with caplog.at_level("WARNING"):
        _assert_scratch_space(Path("/"), 0)
    assert "inactive" in caplog.text


def test_reserve_is_what_separates_fitting_from_safe() -> None:
    """Free space above the estimate but inside the reserve must still be refused."""
    need = VAL_FRAMES * _BYTES_PER_FRAME
    just_over = (need + _MIN_RESERVE // 2) / 2**30
    with mock.patch("shutil.disk_usage", return_value=_usage(50, just_over)), _on_device(1):
        with pytest.raises(RuntimeError):
            _assert_scratch_space(Path("/tmp"), VAL_FRAMES)


@pytest.mark.parametrize("layout", ["sequences", "dataset/sequences"])
def test_count_frames_handles_both_dataset_layouts(tmp_path: Path, layout: str) -> None:
    """Both layouts ship in the wild; guessing one yields 0 frames and a dead guard."""
    seq = tmp_path / "SemanticKITTI" / layout / "08" / "velodyne"
    seq.mkdir(parents=True)
    for i in range(7):
        (seq / f"{i:06d}.bin").touch()
    assert _count_frames(tmp_path, "08") == 7


def test_count_frames_sums_sequences_and_skips_missing(tmp_path: Path) -> None:
    for seq_id, n in (("08", 5), ("09", 3)):
        d = tmp_path / "SemanticKITTI" / "sequences" / seq_id / "velodyne"
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i:06d}.bin").touch()
    assert _count_frames(tmp_path, "08,09") == 8
    assert _count_frames(tmp_path, "08,99") == 5      # missing contributes 0, does not raise


def test_counts_the_shipped_val_sequence_if_present() -> None:
    """Live check against the real data root, skipped where the dataset is absent."""
    root = Path(__file__).resolve().parent.parent / "data"
    if not (root / "SemanticKITTI" / "sequences" / "08" / "velodyne").is_dir():
        pytest.skip("SemanticKITTI not present in this checkout")
    assert _count_frames(root, "08") == VAL_FRAMES
