import numpy as np

from dynacell.geometry import (
    find_inscribed_bbox,
    find_overlap_bbox_across_time,
)
from dynacell.run import run_from_config


def test_overlap_bbox():
    """Test with synthetic numpy arrays."""
    T, C_lf, C_ls, Z, Y, X = 3, 2, 1, 5, 100, 120
    rng = np.random.default_rng(42)

    # Label-free: zero padding on the left, increasing per t
    lf = rng.random((T, C_lf, Z, Y, X), dtype=np.float32) + 0.1
    for t in range(T):
        lf[t, :, :, :, : 10 + t] = 0

    # Light-sheet: zero padding on the right, increasing per t
    ls = rng.random((T, C_ls, Z, Y, X), dtype=np.float32) + 0.1
    for t in range(T):
        ls[t, :, :, :, -(15 + t) :] = 0

    # Run
    result = find_overlap_bbox_across_time([lf, ls])
    assert result is not None, "Expected overlap, got None"
    (y_min, y_max, x_min, x_max), mask, per_t = result

    # Expected per-t:
    #   t=0: lf zeros cols 0-9,  ls zeros cols 105-119 -> X overlap [10, 104]
    #   t=1: lf zeros cols 0-10, ls zeros cols 104-119 -> X overlap [11, 103]
    #   t=2: lf zeros cols 0-11, ls zeros cols 103-119 -> X overlap [12, 102]
    # Intersection: x_min=12, x_max=102, y_min=0, y_max=99
    assert y_min == 0, f"y_min={y_min} != 0"
    assert y_max == 99, f"y_max={y_max} != 99"
    assert x_min == 12, f"x_min={x_min} != 12"
    assert x_max == 102, f"x_max={x_max} != 102"
    assert mask.shape == (100, 120), f"mask shape={mask.shape}"
    assert mask[50, 50] is np.True_, "center should be valid"
    assert mask[50, 0] is np.False_, "left edge should be invalid"

    print(f"Overlap bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print(f"Crop size: Y={y_max - y_min + 1}, X={x_max - x_min + 1}")
    print("TEST PASSED!")


def test_overlap_bbox_no_overlap():
    """Test with arrays that have no overlapping non-zero region."""
    T, C, Z, Y, X = 2, 1, 3, 50, 50
    rng = np.random.default_rng(0)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[:, :, :, :, 25:] = 0  # only left half

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[:, :, :, :, :25] = 0  # only right half

    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is None, f"Expected None, got {result}"
    print("NO-OVERLAP TEST PASSED!")


def test_overlap_bbox_with_y_padding():
    """Test with zero padding in both Y and X."""
    T, C, Z, Y, X = 2, 1, 5, 80, 100
    rng = np.random.default_rng(7)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[:, :, :, :5, :] = 0   # top 5 rows zero
    arr1[:, :, :, :, :8] = 0   # left 8 cols zero

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[:, :, :, 70:, :] = 0  # bottom 10 rows zero
    arr2[:, :, :, :, 90:] = 0  # right 10 cols zero

    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is not None
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert y_min == 5, f"y_min={y_min} != 5"
    assert y_max == 69, f"y_max={y_max} != 69"
    assert x_min == 8, f"x_min={x_min} != 8"
    assert x_max == 89, f"x_max={x_max} != 89"
    print(f"Y+X padding bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("Y+X PADDING TEST PASSED!")


def test_overlap_bbox_different_spatial_dims():
    """Test with arrays that have different Y,X dimensions (like real data)."""
    T, Z = 2, 5
    rng = np.random.default_rng(99)

    # arr1: (T, 1, Z, 100, 80) - zeros on left 5 cols
    arr1 = rng.random((T, 1, Z, 100, 80), dtype=np.float32) + 0.1
    arr1[:, :, :, :, :5] = 0

    # arr2: (T, 2, Z, 60, 120) - zeros on bottom 10 rows
    arr2 = rng.random((T, 2, Z, 60, 120), dtype=np.float32) + 0.1
    arr2[:, :, :, 50:, :] = 0

    # Common region: Y=min(100,60)=60, X=min(80,120)=80
    # arr1 in common: zeros cols 0-4, valid Y=[0:60], X=[5:80]
    # arr2 in common: zeros rows 50-59, valid Y=[0:50], X=[0:80]
    # Overlap: Y=[0:50], X=[5:80]
    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is not None, f"Expected overlap, got None"
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert y_min == 0, f"y_min={y_min} != 0"
    assert y_max == 49, f"y_max={y_max} != 49"
    assert x_min == 5, f"x_min={x_min} != 5"
    assert x_max == 79, f"x_max={x_max} != 79"
    # mask shape should be min(Y), min(X) = 60, 80
    assert mask.shape == (60, 80), f"mask shape={mask.shape}"
    print(f"Different dims bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("DIFFERENT SPATIAL DIMS TEST PASSED!")


def test_overlap_bbox_with_nans():
    """Test that NaN regions are treated as invalid (like zeros)."""
    T, C, Z, Y, X = 2, 1, 3, 50, 60
    rng = np.random.default_rng(13)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[:, :, :, :, :10] = np.nan  # NaN on left

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[:, :, :, :, 50:] = 0  # zeros on right

    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is not None
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert x_min == 10, f"x_min={x_min} != 10"
    assert x_max == 49, f"x_max={x_max} != 49"
    print(f"NaN bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("NAN TEST PASSED!")


def test_blank_first_frame():
    """Test that a blank first timepoint is properly skipped."""
    T, C, Z, Y, X = 5, 1, 5, 50, 60
    rng = np.random.default_rng(77)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[0, :, :, :, :] = 0  # t=0 completely blank
    arr1[:, :, :, :, :5] = 0  # left 5 cols always zero

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[0, :, :, :, :] = 0  # t=0 completely blank
    arr2[:, :, :, :, 50:] = 0  # right 10 cols always zero

    # Without skip_frames: blank t=0 poisons combined_mask -> None
    result_no_skip = find_overlap_bbox_across_time([arr1, arr2])
    assert result_no_skip is None, "Without skip_frames, blank t=0 should cause no overlap"

    # With skip_frames: blank t=0 is excluded
    result = find_overlap_bbox_across_time([arr1, arr2], skip_frames=[0])
    assert result is not None, "With skip_frames=[0], should find valid overlap"
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert x_min == 5, f"x_min={x_min} != 5"
    assert x_max == 49, f"x_max={x_max} != 49"
    assert per_t[0].tolist() == [0, 0, 0, 0], "Blank frame should have zero bbox"
    assert per_t[1, 2] > 0, "Non-blank frame should have valid x_min"

    print(f"Blank t=0 bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("BLANK FIRST FRAME TEST PASSED!")


def test_inscribed_bbox():
    """Test find_inscribed_bbox on a mildly sheared parallelogram (realistic LS data)."""
    # 100x120 mask with mild shear: ~0.1 px shift per row (10px total over 100 rows)
    # This matches real deskewed LS data where shear is small relative to width.
    mask = np.zeros((100, 120), dtype=bool)
    width = 80
    for y in range(100):
        x_start = y // 10  # shifts right by 1 every 10 rows
        mask[y, x_start : x_start + width] = True

    # Bounding box would be (0, 99, 0, 89) — includes invalid sheared corners.
    # Border-shrink should find a rectangle fully inside the parallelogram.
    result = find_inscribed_bbox(mask)
    assert result is not None
    y_min, y_max, x_min, x_max = result
    # Verify all pixels in the bbox are True
    assert mask[y_min : y_max + 1, x_min : x_max + 1].all(), "Bbox has False pixels!"
    # Should keep most of the area (mild shear loses only ~10 cols)
    area = (y_max - y_min + 1) * (x_max - x_min + 1)
    assert area >= 80 * 70, f"Area {area} too small for mild shear"
    print(f"Inscribed bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}], area={area}")
    print("INSCRIBED BBOX TEST PASSED!")


def test_overlap_bbox_sheared():
    """Test that sheared LS overlap is handled via inscribed bbox."""
    T, C, Z, Y, X = 2, 1, 5, 100, 120
    rng = np.random.default_rng(55)

    # LF: valid everywhere
    lf = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1

    # LS: mildly sheared parallelogram (realistic: ~0.1 px shift per row)
    ls = np.zeros((T, C, Z, Y, X), dtype=np.float32)
    for y in range(Y):
        x_start = y // 10  # 10px total shift over 100 rows
        x_end = x_start + 80
        if x_end <= X:
            ls[:, :, :, y, x_start:x_end] = (
                rng.random((T, C, Z, x_end - x_start), dtype=np.float32) + 0.1
            )

    result = find_overlap_bbox_across_time([lf, ls])
    assert result is not None
    (y_min, y_max, x_min, x_max), mask, per_t = result

    # The bbox must be fully inside the valid LS region
    for y in range(y_min, y_max + 1):
        ls_start = y // 10
        assert x_min >= ls_start, (
            f"Row {y}: x_min={x_min} < ls_start={ls_start}"
        )
        assert x_max < ls_start + 80, (
            f"Row {y}: x_max={x_max} >= ls_end={ls_start + 80}"
        )

    area = (y_max - y_min + 1) * (x_max - x_min + 1)
    print(
        f"Sheared bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}], area={area}"
    )
    print("SHEARED OVERLAP TEST PASSED!")


if __name__ == "__main__":
    test_inscribed_bbox()
    test_overlap_bbox()
    test_overlap_bbox_no_overlap()
    test_overlap_bbox_with_y_padding()
    test_overlap_bbox_different_spatial_dims()
    test_overlap_bbox_with_nans()
    test_blank_first_frame()
    test_overlap_bbox_sheared()
    print("\n=== SUBMITTING ALL FOVs ===")



    # root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    # dataset = "2025_08_26_A549_SEC61_TOMM20_ZIKV"
    # run_all_fovs(
    #     root_path=root_path,
    #     dataset=dataset,
    #     lf_mask_radius=0.98,
    #     local=False,
    #     stage1_run_dir=None,
    #     beads_fov="A/3/000000",
    #     overlay_channels=["Phase3D", "raw GFP EX488 EM525-45"],
    #     exclude_fovs=["A/3/000001", "A/3/001000", "A/3/001001"],
    #     z_final = 48,
    #     n_std = 2.5,
    #     z_window = 20,
    # )

    run_from_config("configs/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.yaml")

    # 2024_11_05_A549_TOMM20_ZIKV_DENV

    # root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    # dataset = "2024_11_05_A549_TOMM20_ZIKV_DENV"
    # run_all_fovs(
    #     root_path=root_path,
    #     dataset=dataset,
    #     lf_mask_radius=0.98,
    #     local=False,
    #     stage1_run_dir=None,
    #     beads_fov="C/1/000000",
    #     overlay_channels=["Phase3D", "raw GFP EX488 EM525-45"],
    #     #exclude_fovs=["A/3/000001", "A/3/001000", "A/3/001001"],
    #     z_final = 48,
    #     n_std = 2.5,
    #     #z_window = 20,
    # )

    # root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    # dataset = "2024_11_07_A549_SEC61_DENV"
    # run_all_fovs(
    #     root_path=root_path,
    #     dataset=dataset,
    #     lf_mask_radius=0.98,
    #     local=False,
    #     stage1_run_dir=None,
    #     beads_fov="C/1/000000",
    #     overlay_channels=["Phase3D", "raw GFP EX488 EM525-45"],
    #     #exclude_fovs=["A/3/000001", "A/3/001000", "A/3/001001"],
    #     z_final = 48,
    #     n_std = 2.5,
    #     z_window = 20,
    # )

    # root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    # dataset = "2024_12_05_A549_LAMP1_ZIKV_DENV"
    # run_all_fovs(
    #     root_path=root_path,
    #     dataset=dataset,
    #     lf_mask_radius=0.98,
    #     local=False,
    #     stage1_run_dir=None,
    #     beads_fov="C/1/000000",
    #     overlay_channels=["Phase3D", "raw GFP EX488 EM525-45"],
    #     #exclude_fovs=["A/3/000001", "A/3/001000", "A/3/001001"],
    #     z_final = 48,
    #     n_std = 2.5,
    #     z_window = 20,
    # )


    # root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    # dataset = "2024_10_31_A549_SEC61_ZIKV_DENV"
    # run_all_fovs(
    #     root_path=root_path,
    #     dataset=dataset,
    #     lf_mask_radius=0.98,
    #     local=False,
    #     stage1_run_dir=None,
    #     beads_fov="C/1/000000",
    #     overlay_channels=["Phase3D", "raw GFP EX488 EM525-45"],
    #  #   exclude_fovs=["A/3/000000", "A/3/001000", "A/3/001001"],
    #     z_final = 48,
    #     n_std = 3.5,
    #     z_window = 20,
    # )


    # root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    # dataset = "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
    # run_all_fovs(
    #     root_path=root_path,
    #     dataset=dataset,
    #     lf_mask_radius=0.98,
    #     local=False,
    #     stage1_run_dir=None,
    #     beads_fov="A/3/000001",
    #     overlay_channels=["Phase3D", "raw GFP EX488 EM525-45"],
    #     exclude_fovs=["A/3/000000", "A/3/001000", "A/3/001001"],
    #     z_final = 48,
    #     n_std = 2,
    #     z_window = 20,
    #     stages=[1, 2],
    # )
