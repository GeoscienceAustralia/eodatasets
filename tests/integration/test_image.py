import numpy as np

from eodatasets3 import images


def test_rescale_intensity():
    # Example was generated via:
    #     scipy.ndimage.rotate(np.arange(1000, 8000, 100).reshape((7,10)), 45, cval=-99)

    # (Using a variable so the array is more spaced-out & readable)
    nada = -999
    original_image = np.array(
        [
            [nada, nada, nada, nada, nada, nada, nada, nada, nada, nada, nada, nada],
            [nada, nada, nada, nada, nada, nada, 1852, 2730, nada, nada, nada, nada],
            [nada, nada, nada, nada, nada, 1711, 2570, 3428, 4169, nada, nada, nada],
            [nada, nada, nada, nada, 1568, 2432, 3287, 4009, 4805, 5610, nada, nada],
            [nada, nada, nada, 1427, 2291, 3144, 3871, 4663, 5451, 6181, 7049, nada],
            [nada, nada, 1284, 2149, 3003, 3729, 4521, 5312, 6040, 6889, 7757, nada],
            [nada, 1143, 2011, 2860, 3588, 4379, 5171, 5897, 6751, 7616, nada, nada],
            [nada, 1851, 2719, 3449, 4237, 5029, 5756, 6609, 7473, nada, nada, nada],
            [nada, nada, 3290, 4095, 4891, 5613, 6468, 7332, nada, nada, nada, nada],
            [nada, nada, nada, 4731, 5472, 6330, 7189, nada, nada, nada, nada, nada],
            [nada, nada, nada, nada, 6170, 7048, nada, nada, nada, nada, nada, nada],
            [nada, nada, nada, nada, nada, nada, nada, nada, nada, nada, nada, nada],
        ]
    )
    unmodified = original_image.copy()

    assert np.array_equal(original_image, unmodified), (
        "rescale_intensity modified the input image"
    )

    staticly_rescaled = images.rescale_intensity(
        original_image, in_range=(4000, 6000), out_range=(100, 255), image_nodata=-999
    )
    print("Statically rescaled result: ")
    print(repr(staticly_rescaled))

    # - Note that the nodata values are not scaled (a previous bug!)
    #   they're translated to the output nodata value (0).
    # - Note how many will be clipped to the min (100) without falling into nodata.
    non = 0
    expected_static_rescale = np.array(
        [
            [non, non, non, non, non, non, non, non, non, non, non, non],
            [non, non, non, non, non, non, 100, 100, non, non, non, non],
            [non, non, non, non, non, 100, 100, 100, 113, non, non, non],
            [non, non, non, non, 100, 100, 100, 100, 162, 224, non, non],
            [non, non, non, 100, 100, 100, 100, 151, 212, 255, 255, non],
            [non, non, 100, 100, 100, 100, 140, 201, 255, 255, 255, non],
            [non, 100, 100, 100, 100, 129, 190, 247, 255, 255, non, non],
            [non, 100, 100, 100, 118, 179, 236, 255, 255, non, non, non],
            [non, non, 100, 107, 169, 225, 255, 255, non, non, non, non],
            [non, non, non, 156, 214, 255, 255, non, non, non, non, non],
            [non, non, non, non, 255, 255, non, non, non, non, non, non],
            [non, non, non, non, non, non, non, non, non, non, non, non],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(staticly_rescaled, expected_static_rescale)


def test_calc_range():
    # Test that the correct value range and valid data arrays are calculated.

    # Test arrays generated via:
    # >>> scipy.ndimage.rotate(np.arange(10, 70, 1).reshape((6, 10)), 55, cval=-11)
    # >>> scipy.ndimage.rotate(np.arange(20, 80, 1).reshape((6, 10)), 50, cval=-11)
    # >>> scipy.ndimage.rotate(np.arange(30, 90, 1).reshape((6, 10)), 55, cval=-11)

    # They have:
    # - slightly different values to test the highest/lowest value range calculation
    #   (it should be across all bands)
    # - And slightly different rotation to test the combined valid_data mask.

    no = -11
    r_array = np.array(
        [
            [no, no, no, no, no, no, no, no, no, no, no],
            [no, no, no, no, no, no, 25, no, no, no, no],
            [no, no, no, no, no, 21, 31, 40, no, no, no],
            [no, no, no, no, 17, 27, 36, 45, 53, 64, no],
            [no, no, no, 15, 23, 32, 41, 49, 59, 68, no],
            [no, no, no, 18, 29, 37, 46, 54, 65, no, no],
            [no, no, 14, 25, 33, 42, 50, 61, no, no, no],
            [no, 11, 20, 30, 38, 47, 56, 64, no, no, no],
            [no, 15, 26, 34, 43, 52, 62, no, no, no, no],
            [no, no, no, 39, 48, 58, no, no, no, no, no],
            [no, no, no, no, 54, no, no, no, no, no, no],
            [no, no, no, no, no, no, no, no, no, no, no],
        ]
    )
    g_array = np.array(
        [
            [no, no, no, no, no, no, no, no, no, no, no],
            [no, no, no, no, no, no, 31, no, no, no, no],
            [no, no, no, no, no, 28, 38, 47, no, no, no],
            [no, no, no, no, 26, 35, 44, 52, 60, 68, no],
            [no, no, no, 24, 32, 41, 49, 58, 66, 76, no],
            [no, no, no, 29, 39, 47, 55, 63, 73, no, no],
            [no, no, 26, 36, 44, 52, 60, 70, no, no, no],
            [no, 23, 33, 41, 50, 58, 67, 75, no, no, no],
            [no, 31, 39, 47, 55, 64, 73, no, no, no, no],
            [no, no, no, 52, 61, 71, no, no, no, no, no],
            [no, no, no, no, 68, no, no, no, no, no, no],
            [no, no, no, no, no, no, no, no, no, no, no],
        ]
    )
    b_array = np.array(
        [
            [no, no, no, no, no, no, no, no, no, no, no],
            [no, no, no, no, no, no, 45, no, no, no, no],
            [no, no, no, no, no, 41, 51, 60, no, no, no],
            [no, no, no, no, 37, 47, 56, 65, 73, 84, no],
            [no, no, no, 35, 43, 52, 61, 69, 79, 88, no],
            [no, no, no, 38, 49, 57, 66, 74, 85, no, no],
            [no, no, 34, 45, 53, 62, 70, 81, no, no, no],
            [no, 31, 40, 50, 58, 67, 76, 84, no, no, no],
            [no, 35, 46, 54, 63, 72, 82, no, no, no, no],
            [no, no, no, 59, 68, 78, no, no, no, no, no],
            [no, no, no, no, 74, no, no, no, no, no, no],
            [no, no, no, no, no, no, no, no, no, no, no],
        ]
    )

    mask = np.ones(r_array.shape, dtype=np.bool_)
    calculated_range = images.read_valid_mask_and_value_range(
        mask,
        ((r_array, no), (g_array, no), (b_array, no)),
        calculate_percentiles=(2, 98),
    )

    expected_combined_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert np.array_equal(expected_combined_mask, mask), (
        f"Combined mask isn't as expected. "
        f"Diff: {np.logical_xor(expected_combined_mask, mask)!r}"
    )

    assert calculated_range == (
        34,
        65,
    ), f"Unexpected 2/98 percentile values: {calculated_range}"
