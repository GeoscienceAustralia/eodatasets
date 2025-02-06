import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from eodatasets3.prepare import landsat_l1_prepare
from tests.common import check_prepare_outputs, run_prepare_cli

LC08_L2_C2_POST_20210507_INPUT_PATH: Path = (
    Path(__file__).parent.parent / "data" / "LC08_L2SP_098084_20210503_20210508_02_T1"
)

LT05_L2_C2_INPUT_PATH: Path = (
    Path(__file__).parent.parent / "data" / "LT05_L2SP_090084_19980308_20200909_02_T1"
)

LE07_L2_C2_INPUT_PATH: Path = (
    Path(__file__).parent.parent / "data" / "LE07_L2SP_090084_20210331_20210426_02_T1"
)

LE07_L1_C2_INPUT_PATH: Path = (
    Path(__file__).parent.parent / "data" / "LE07_L1TP_107068_20220310_20220405_02_T1"
)


@pytest.fixture
def lc08_l2_c2_post_20210507_folder(tmp_path: Path) -> Path:
    return _make_copy(LC08_L2_C2_POST_20210507_INPUT_PATH, tmp_path)


@pytest.fixture
def lt05_l2_c2_folder(tmp_path: Path) -> Path:
    return _make_copy(LT05_L2_C2_INPUT_PATH, tmp_path)


@pytest.fixture
def le07_l2_c2_folder(tmp_path: Path) -> Path:
    return _make_copy(LE07_L2_C2_INPUT_PATH, tmp_path)


@pytest.fixture
def le07_l1_c2_folder(tmp_path: Path) -> Path:
    return _make_copy(LE07_L1_C2_INPUT_PATH, tmp_path)


def relative_offset(base, offset):
    return offset


def _make_copy(input_path, tmp_path):
    our_input = tmp_path / input_path.name
    if input_path.is_file():
        shutil.copy(input_path, our_input)
    else:
        shutil.copytree(input_path, our_input)
    return our_input


def test_prepare_l5_l1_usgs_tarball(
    tmp_path: Path, l1_ls5_tarball_md_expected: dict, l1_ls5_tarball: Path
):
    assert l1_ls5_tarball.exists(), "Test data missing(?)"
    output_path: Path = tmp_path / "out"
    output_path.mkdir()

    # When specifying an output base path it will create path/row subfolders within it.
    expected_metadata_path = (
        output_path
        / "090"
        / "085"
        / "LT05_L1TP_090085_19970406_20161231_01_T1.odc-metadata.yaml"
    )

    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=["--output-base", str(output_path), str(l1_ls5_tarball)],
        expected_doc=l1_ls5_tarball_md_expected,
        expected_metadata_path=expected_metadata_path,
    )


def test_prepare_l8_l1_usgs_tarball(l1_ls8_folder, l1_ls8_folder_md_expected):
    assert l1_ls8_folder.exists(), "Test data missing(?)"

    # No output path defined,so it will default to being a sibling to the input.
    expected_metadata_path = (
        l1_ls8_folder.parent
        / "LC08_L1TP_090084_20160121_20170405_01_T1.odc-metadata.yaml"
    )
    assert not expected_metadata_path.exists()

    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[str(l1_ls8_folder)],
        expected_doc=l1_ls8_folder_md_expected,
        expected_metadata_path=expected_metadata_path,
    )


def test_prepare_l8_l1_c2(tmp_path: Path, l1_c2_ls8_folder: Path):
    """Run prepare script with a source telemetry data and unique producer."""
    assert l1_c2_ls8_folder.exists(), "Test data missing(?)"

    output_path = tmp_path
    expected_metadata_path = (
        output_path
        / "089"
        / "074"
        / "LC08_L1GT_089074_20220506_20220512_02_T2.odc-metadata.yaml"
    )

    expected_doc = {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": "5e1359de-0f91-5988-a6f0-8a0a840906f4",
        "label": "usgs_ls8c_level1_2-0-20220512_089074_2022-05-06",
        "product": {
            "name": "usgs_ls8c_level1_2",
            "href": "https://collections.dea.ga.gov.au/product/usgs_ls8c_level1_2",
        },
        "crs": "epsg:32656",
        "grids": {
            "default": {
                "shape": [60, 60],
                "transform": [
                    3860.5,
                    0.0,
                    594285.0,
                    0.0,
                    -3910.5,
                    -2121285.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
            "panchromatic": {
                "shape": [60, 60],
                "transform": [
                    3860.2500000000005,
                    0.0,
                    594292.5,
                    0.0,
                    -3910.25,
                    -2121292.5,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": (
                (
                    (635871.100911986, -2328431.8044824763),
                    (598807.3537167514, -2315758.4765276713),
                    (594285.0, -2314216.125490317),
                    (594285.0, -2294479.549612592),
                    (629132.1230216483, -2128210.3416854665),
                    (634001.1369104853, -2121285.0),
                    (671771.6827861228, -2121285.0),
                    (759108.8670950294, -2143693.8957929504),
                    (819064.1631236447, -2156587.0374096073),
                    (824424.9683929395, -2160452.642015061),
                    (825915.0, -2160834.95297774),
                    (825915.0, -2197333.359107589),
                    (818095.1372109791, -2231658.3131009457),
                    (784506.5705934291, -2355915.0),
                    (764094.013739199, -2355915.0),
                    (647454.7800689514, -2332349.7060506954),
                    (636125.9731110331, -2328486.0106703877),
                    (635871.100911986, -2328431.8044824763),
                ),
            ),
        },
        "properties": {
            "datetime": "2022-05-06T23:39:59.285133",
            "eo:cloud_cover": 86.35,
            "eo:gsd": 15.0,
            "eo:instrument": "OLI_TIRS",
            "eo:platform": "landsat-8",
            "eo:sun_azimuth": 39.80724521,
            "eo:sun_elevation": 43.24426868,
            "landsat:collection_category": "T2",
            "landsat:collection_number": 2,
            "landsat:data_type": "L1GT",
            "landsat:landsat_product_id": "LC08_L1GT_089074_20220506_20220512_02_T2",
            "landsat:landsat_scene_id": "LC80890742022126LGN00",
            "landsat:processing_software_version": "LPGS_15.6.0",
            "landsat:station_id": "LGN",
            "landsat:wrs_path": 89,
            "landsat:wrs_row": 74,
            "odc:dataset_version": "2.0.20220512",
            "odc:file_format": "GeoTIFF",
            "odc:processing_datetime": "2022-05-12T14:00:17",
            "odc:producer": "usgs.gov",
            "odc:product_family": "level1",
            "odc:region_code": "089074",
        },
        "measurements": {
            "blue": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B2.TIF"},
            "cirrus": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B9.TIF"},
            "coastal_aerosol": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_B1.TIF"
            },
            "green": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B3.TIF"},
            "lwir_1": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B10.TIF"},
            "lwir_2": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B11.TIF"},
            "nir": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B5.TIF"},
            "panchromatic": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_B8.TIF",
                "grid": "panchromatic",
            },
            "qa_radsat": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_QA_RADSAT.TIF"
            },
            "quality": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_QA_PIXEL.TIF"
            },
            "red": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B4.TIF"},
            "solar_azimuth": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_SAA.TIF"
            },
            "solar_zenith": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_SZA.TIF"
            },
            "swir_1": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B6.TIF"},
            "swir_2": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_B7.TIF"},
            "view_azimuth": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_VAA.TIF"
            },
            "view_zenith": {"path": "LC08_L1GT_089074_20220506_20220512_02_T2_VZA.TIF"},
        },
        "accessories": {
            "metadata:landsat_mtl": {
                "path": "LC08_L1GT_089074_20220506_20220512_02_T2_MTL.txt"
            }
        },
        "lineage": {},
    }

    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[
            "--output-base",
            output_path,
            "--producer",
            "usgs.gov",
            l1_c2_ls8_folder,
        ],
        expected_doc=expected_doc,
        expected_metadata_path=expected_metadata_path,
    )


@pytest.fixture
def l9_expected():
    return {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": "dba08e46-2651-5125-b811-c4adf0a8e450",
        "label": "usgs_ls9c_level1_2-0-20220209_112081_2022-02-09",
        "product": {
            "name": "usgs_ls9c_level1_2",
            "href": "https://collections.dea.ga.gov.au/product/usgs_ls9c_level1_2",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": (
                (
                    (384592.5, -3430919.6807745784),
                    (384592.5, -3430453.8689052663),
                    (384585.0, -3430451.9965122077),
                    (384585.0, -3393908.1584824338),
                    (419428.36278902093, -3239400.6840891885),
                    (422756.67805196694, -3236385.0),
                    (462566.30393798096, -3236385.0),
                    (616215.0, -3274336.4508977565),
                    (616215.0, -3302620.079207405),
                    (616207.5, -3302673.174815242),
                    (616207.5, -3307542.583714816),
                    (614513.9386270659, -3314662.5975493155),
                    (612262.5701458402, -3330600.9679461434),
                    (584927.4336458556, -3439049.187930595),
                    (577611.2836455577, -3469807.5),
                    (577174.5847835429, -3469807.5),
                    (577172.6943557998, -3469815.0),
                    (542256.2351088949, -3469815.0),
                    (528099.5549533233, -3466280.7507125223),
                    (384592.5, -3430919.6807745784),
                ),
            ),
        },
        "crs": "epsg:32650",
        "grids": {
            "default": {
                "shape": [60, 60],
                "transform": [
                    3860.5,
                    0.0,
                    384585.0,
                    0.0,
                    -3890.5,
                    -3236385.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
            "panchromatic": {
                "shape": [60, 60],
                "transform": [
                    3860.2500000000005,
                    0.0,
                    384592.5,
                    0.0,
                    -3890.2500000000005,
                    -3236392.5,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        },
        "properties": {
            "datetime": "2022-02-09T02:05:18.736033",
            "eo:cloud_cover": 0.12,
            "eo:gsd": 15.0,
            "eo:instrument": "OLI_TIRS",
            "eo:platform": "landsat-9",
            "eo:sun_azimuth": 72.16674497,
            "eo:sun_elevation": 54.14346217,
            "landsat:collection_category": "T1",
            "landsat:collection_number": 2,
            "landsat:data_type": "L1TP",
            "landsat:geometric_rmse_model_x": 4.208,
            "landsat:geometric_rmse_model_y": 4.237,
            "landsat:ground_control_points_model": 857,
            "landsat:ground_control_points_version": 5,
            "landsat:landsat_product_id": "LC09_L1TP_112081_20220209_20220209_02_T1",
            "landsat:landsat_scene_id": "LC91120812022040LGN00",
            "landsat:processing_software_version": "LPGS_15.6.0",
            "landsat:station_id": "LGN",
            "landsat:wrs_path": 112,
            "landsat:wrs_row": 81,
            "odc:dataset_version": "2.0.20220209",
            "odc:file_format": "GeoTIFF",
            "odc:processing_datetime": "2022-02-09T04:08:31",
            "odc:producer": "usgs.gov",
            "odc:product_family": "level1",
            "odc:region_code": "112081",
        },
        "measurements": {
            "blue": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B2.TIF"},
            "cirrus": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B9.TIF"},
            "coastal_aerosol": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_B1.TIF"
            },
            "green": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B3.TIF"},
            "lwir_1": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B10.TIF"},
            "lwir_2": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B11.TIF"},
            "nir": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B5.TIF"},
            "panchromatic": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_B8.TIF",
                "grid": "panchromatic",
            },
            "qa_radsat": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_QA_RADSAT.TIF"
            },
            "quality": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_QA_PIXEL.TIF"
            },
            "red": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B4.TIF"},
            "solar_azimuth": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_SAA.TIF"
            },
            "solar_zenith": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_SZA.TIF"
            },
            "swir_1": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B6.TIF"},
            "swir_2": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_B7.TIF"},
            "view_azimuth": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_VAA.TIF"
            },
            "view_zenith": {"path": "LC09_L1TP_112081_20220209_20220209_02_T1_VZA.TIF"},
        },
        "accessories": {
            "metadata:landsat_mtl": {
                "path": "LC09_L1TP_112081_20220209_20220209_02_T1_MTL.txt"
            }
        },
        "lineage": {},
    }


def test_prepare_l9_l1_c2(tmp_path: Path, l1_ls9_tarball: Path, l9_expected: dict):
    """Run prepare script with a source telemetry data and unique producer."""
    assert l1_ls9_tarball.exists(), "Test data missing(?)"

    output_path = tmp_path
    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[
            "--output-base",
            output_path,
            "--producer",
            "usgs.gov",
            l1_ls9_tarball,
        ],
        expected_doc=l9_expected,
        expected_metadata_path=(
            output_path
            / "112"
            / "081"
            / "LC09_L1TP_112081_20220209_20220209_02_T1.odc-metadata.yaml"
        ),
    )


def test_prepare_lc08_l2_c2_post_20210507(
    tmp_path: Path,
    lc08_l2_c2_post_20210507_folder: Path,
):
    """Support a functionality baseline for the enhancements to expand landsat
    prepare (YAML) logic to support USGS level 2 - PR#159:
     LC08 C2 L2 post 7th May 2021."""
    assert lc08_l2_c2_post_20210507_folder.exists(), "Test data missing(?)"

    output_path = tmp_path
    expected_metadata_path = (
        output_path
        / "098"
        / "084"
        / "LC08_L2SP_098084_20210503_20210508_02_T1.odc-metadata.yaml"
    )
    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[
            "--output-base",
            output_path,
            "--producer",
            "usgs.gov",
            lc08_l2_c2_post_20210507_folder,
        ],
        expected_doc=expected_lc08_l2_c2_post_20210507_folder(),
        expected_metadata_path=expected_metadata_path,
    )


def test_prepare_lt05_l2_c2(
    tmp_path: Path,
    lt05_l2_c2_folder: Path,
):
    """Support a functionality baseline for the enhancements to expand landsat
    prepare (YAML) logic to support USGS level 2 - PR#159:
     LT05 C2 L2."""
    assert lt05_l2_c2_folder.exists(), "Test data missing(?)"

    output_path = tmp_path
    expected_metadata_path = (
        output_path
        / "090"
        / "084"
        / "LT05_L2SP_090084_19980308_20200909_02_T1.odc-metadata.yaml"
    )
    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[
            "--output-base",
            output_path,
            "--producer",
            "usgs.gov",
            lt05_l2_c2_folder,
        ],
        expected_doc=expected_lt05_l2_c2_folder(),
        expected_metadata_path=expected_metadata_path,
    )


def test_prepare_le07_l2_c2(
    tmp_path: Path,
    le07_l2_c2_folder: Path,
):
    """Support a functionality baseline for the enhancements to expand landsat
    prepare (YAML) logic to support USGS level 2 - PR#159:
     LE07 C2 L2."""
    assert le07_l2_c2_folder.exists(), "Test data missing(?)"

    output_path = tmp_path
    expected_metadata_path = (
        output_path
        / "090"
        / "084"
        / "LE07_L2SP_090084_20210331_20210426_02_T1.odc-metadata.yaml"
    )
    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[
            "--output-base",
            output_path,
            "--producer",
            "usgs.gov",
            le07_l2_c2_folder,
        ],
        expected_doc=expected_le07_l2_c2_folder(),
        expected_metadata_path=expected_metadata_path,
    )


def test_prepare_le07_l1_c2(
    tmp_path: Path,
    le07_l1_c2_folder: Path,
):
    assert le07_l1_c2_folder.exists(), "Test data missing(?)"

    output_path = tmp_path
    expected_metadata_path = (
        output_path
        / "107"
        / "068"
        / "LE07_L1TP_107068_20220310_20220405_02_T1.odc-metadata.yaml"
    )
    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[
            "--output-base",
            output_path,
            "--producer",
            "usgs.gov",
            le07_l1_c2_folder,
        ],
        expected_doc=expected_le07_l1_c2_folder(),
        expected_metadata_path=expected_metadata_path,
    )


def test_prepare_l8_l1_tarball_with_source(
    tmp_path: Path, l1_ls8_folder: Path, ls8_telemetry_path, l1_ls8_ga_expected: dict
):
    """Run prepare script with a source telemetry data and unique producer."""
    assert l1_ls8_folder.exists(), "Test data missing(?)"

    output_path = tmp_path
    expected_metadata_path = (
        output_path
        / "090"
        / "084"
        / "LC08_L1TP_090084_20160121_20170405_01_T1.odc-metadata.yaml"
    )
    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[
            "--output-base",
            output_path,
            "--producer",
            "ga.gov.au",
            "--source",
            ls8_telemetry_path,
            l1_ls8_folder,
        ],
        expected_doc=l1_ls8_ga_expected,
        expected_metadata_path=expected_metadata_path,
    )


def test_prepare_l7_l1_usgs_tarball(
    l1_ls7_tarball: Path, l1_ls7_tarball_md_expected: dict
):
    assert l1_ls7_tarball.exists(), "Test data missing(?)"

    expected_metadata_path = (
        l1_ls7_tarball.parent
        / "LE07_L1TP_104078_20130429_20161124_01_T1.odc-metadata.yaml"
    )

    check_prepare_outputs(
        invoke_script=landsat_l1_prepare.main,
        run_args=[str(l1_ls7_tarball)],
        expected_doc=l1_ls7_tarball_md_expected,
        expected_metadata_path=expected_metadata_path,
    )


def test_skips_old_datasets(l1_ls7_tarball):
    """Prepare should skip datasets older than the given date"""
    expected_metadata_path = (
        l1_ls7_tarball.parent
        / "LE07_L1TP_104078_20130429_20161124_01_T1.odc-metadata.yaml"
    )

    run_prepare_cli(
        landsat_l1_prepare.main,
        # Can't be newer than right now.
        "--newer-than",
        datetime.now().isoformat(),
        str(l1_ls7_tarball),
    )
    assert not expected_metadata_path.exists(), (
        "Dataset should have been skipped due to age"
    )

    # It should work with an old date.
    run_prepare_cli(
        landsat_l1_prepare.main,
        # Some old date, from before the test data was created.
        "--newer-than",
        "2014-05-04",
        str(l1_ls7_tarball),
    )
    assert expected_metadata_path.exists(), (
        "Dataset should have been packaged when using an ancient date cutoff"
    )


def expected_lc08_l2_c2_post_20210507_folder(
    l2_c2_ls8_folder: Path = None,
    offset: Callable[[Path, str], str] = relative_offset,
    organisation="usgs.gov",
    collection="2",
    leveln_collection="2",
    lineage=None,
):
    """ """
    org_code = organisation.split(".")[0]
    product_name = f"{org_code}_ls8c_level{leveln_collection}_{collection}"
    processing_datetime = datetime(2021, 5, 8, 11, 5, 47)
    processing_date = processing_datetime.strftime("%Y%m%d")
    return {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": "d4ff459f-67e0-57df-af73-dfef14ad9c47",
        "label": f"{product_name}-0-{processing_date}_098084_2021-05-03",
        "product": {
            "name": product_name,
            "href": f"https://collections.dea.ga.gov.au/product/{product_name}",
        },
        "properties": {
            "datetime": datetime(2021, 5, 3, 0, 39, 15, 718295),
            # The minor version comes from the processing date,
            # as used in filenames to distinguish reprocesses.
            "odc:dataset_version": f"{collection}.0.{processing_date}",
            "odc:file_format": "GeoTIFF",
            "odc:processing_datetime": processing_datetime,
            "odc:producer": organisation,
            "odc:product_family": "level2",
            "odc:region_code": "098084",
            "eo:cloud_cover": 72.57,
            "eo:gsd": 30.0,
            "eo:instrument": "OLI_TIRS",
            "eo:platform": "landsat-8",
            "eo:sun_azimuth": 36.555_149_01,
            "eo:sun_elevation": 31.263_730_68,
            "landsat:algorithm_source_surface_reflectance": "LaSRC_1.5.0",
            "landsat:collection_category": "T1",
            "landsat:collection_number": int(leveln_collection),
            "landsat:data_type": "L2SP",
            "landsat:geometric_rmse_model_x": 5.974,
            "landsat:geometric_rmse_model_y": 4.276,
            "landsat:ground_control_points_model": 35,
            "landsat:ground_control_points_version": 5,
            "landsat:landsat_product_id": f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1",
            "landsat:landsat_scene_id": "LC80980842021123LGN00",
            "landsat:processing_software_version": "LPGS_15.4.0",
            "landsat:station_id": "LGN",
            "landsat:wrs_path": 98,
            "landsat:wrs_row": 84,
        },
        "crs": "epsg:32653",
        "geometry": {
            "coordinates": [
                [
                    [656390.7366486033, -3713985.0],
                    [653137.3072645832, -3716864.7197616836],
                    [609585.0, -3888976.9699604893],
                    [609585.0, -3905841.19690461],
                    [691402.3702754473, -3932222.5947522246],
                    [777854.1421397077, -3952215.0],
                    [796350.2575455576, -3952215.0],
                    [818563.2799541968, -3885739.54955405],
                    [846315.0, -3761631.0],
                    [823680.1297245529, -3749859.4052477754],
                    [677099.698619789, -3713985.0],
                    [656390.7366486033, -3713985.0],
                ],
            ],
            "type": "Polygon",
        },
        "grids": {
            "default": {
                "shape": (60, 60),
                "transform": (
                    3945.5000000000005,
                    0.0,
                    609585.0,
                    0.0,
                    -3970.5,
                    -3713985.0,
                    0.0,
                    0.0,
                    1.0,
                ),
            },
        },
        "measurements": {
            "atmos_transmittance": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_ATRAN.TIF",
                )
            },
            "blue": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_B2.TIF",
                )
            },
            "cloud_distance": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_CDIST.TIF",
                )
            },
            "coastal_aerosol": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_B1.TIF",
                )
            },
            "downwell_radiance": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_DRAD.TIF",
                )
            },
            "emissivity": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_EMIS.TIF",
                )
            },
            "emissivity_stdev": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_EMSD.TIF",
                )
            },
            "green": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_B3.TIF",
                )
            },
            "lwir": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_B10.TIF",
                )
            },
            "nir": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_B5.TIF",
                )
            },
            "qa_aerosol": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_QA_AEROSOL.TIF",
                )
            },
            "qa_radsat": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_QA_RADSAT.TIF",
                )
            },
            "qa_temperature": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_QA.TIF",
                )
            },
            "quality": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_QA_PIXEL.TIF",
                )
            },
            "red": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_B4.TIF",
                )
            },
            "swir_1": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_B6.TIF",
                )
            },
            "swir_2": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_SR_B7.TIF",
                )
            },
            "thermal_radiance": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_TRAD.TIF",
                )
            },
            "upwell_radiance": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_ST_URAD.TIF",
                )
            },
        },
        "accessories": {
            "metadata:landsat_mtl": {
                "path": offset(
                    l2_c2_ls8_folder,
                    f"LC08_L2SP_098084_20210503_{processing_date}_0{leveln_collection}_T1_MTL.txt",
                )
            }
        },
        "lineage": lineage or {},
    }


def expected_lt05_l2_c2_folder():
    return {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": "b08b3f9e-b00c-5a67-88d8-a889f0e79d00",
        "label": "usgs_ls5t_level2_2-0-20200909_090084_1998-03-08",
        "product": {
            "name": "usgs_ls5t_level2_2",
            "href": "https://collections.dea.ga.gov.au/product/usgs_ls5t_level2_2",
        },
        "properties": {
            "datetime": datetime(1998, 3, 8, 23, 26, 47, 294081),
            # The minor version comes from the processing date,
            # as used in filenames to distinguish reprocesses.
            "odc:dataset_version": "2.0.20200909",
            "odc:file_format": "GeoTIFF",
            "odc:processing_datetime": (datetime(2020, 9, 9, 10, 36, 59)),
            "odc:producer": "usgs.gov",
            "odc:product_family": "level2",
            "odc:region_code": "090084",
            "eo:cloud_cover": 12.0,
            "eo:gsd": 30.0,
            "eo:instrument": "TM",
            "eo:platform": "landsat-5",
            "eo:sun_azimuth": 61.298_799_16,
            "eo:sun_elevation": 41.583_263_99,
            "landsat:algorithm_source_surface_reflectance": "LEDAPS_3.4.0",
            "landsat:collection_category": "T1",
            "landsat:collection_number": 2,
            "landsat:data_type": "L2SP",
            "landsat:geometric_rmse_model_x": 3.085,
            "landsat:geometric_rmse_model_y": 2.977,
            "landsat:ground_control_points_model": 965,
            "landsat:ground_control_points_version": 5,
            "landsat:landsat_product_id": "LT05_L2SP_090084_19980308_20200909_02_T1",
            "landsat:landsat_scene_id": "LT50900841998067ASA00",
            "landsat:processing_software_version": "LPGS_15.3.1c",
            "landsat:station_id": "ASA",
            "landsat:wrs_path": 90,
            "landsat:wrs_row": 84,
        },
        "crs": "epsg:32655",
        "geometry": {
            "coordinates": [
                [
                    [686022.6472422444, -3724785.0],
                    [682595.9899406636, -3727553.91102268],
                    [638085.0, -3907560.0],
                    [790642.5733366499, -3940388.5126599884],
                    [830607.0294154783, -3943044.3288386273],
                    [880215.0, -3761340.0],
                    [707789.0909090909, -3724785.0],
                    [686022.6472422444, -3724785.0],
                ],
            ],
            "type": "Polygon",
        },
        "grids": {
            "default": {
                "shape": (60, 60),
                "transform": (
                    4035.5000000000005,
                    0.0,
                    638085.0,
                    0.0,
                    -3655.5,
                    -3724785.0,
                    0.0,
                    0.0,
                    1.0,
                ),
            },
        },
        "measurements": {
            "atmos_opacity": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_ATMOS_OPACITY.TIF"
            },
            "atmos_transmittance": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_ATRAN.TIF"
            },
            "blue": {"path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_B1.TIF"},
            "cloud_distance": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_CDIST.TIF"
            },
            "downwell_radiance": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_DRAD.TIF"
            },
            "emissivity": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_EMIS.TIF"
            },
            "emissivity_stdev": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_EMSD.TIF"
            },
            "green": {"path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_B2.TIF"},
            "lwir": {"path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_B6.TIF"},
            "nir": {"path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_B4.TIF"},
            "qa_cloud": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_CLOUD_QA.TIF"
            },
            "qa_radsat": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_QA_RADSAT.TIF"
            },
            "qa_temperature": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_QA.TIF"
            },
            "quality": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_QA_PIXEL.TIF"
            },
            "red": {"path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_B3.TIF"},
            "swir_1": {"path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_B5.TIF"},
            "swir_2": {"path": "LT05_L2SP_090084_19980308_20200909_02_T1_SR_B7.TIF"},
            "thermal_radiance": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_TRAD.TIF"
            },
            "upwell_radiance": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_ST_URAD.TIF"
            },
        },
        "accessories": {
            "metadata:landsat_mtl": {
                "path": "LT05_L2SP_090084_19980308_20200909_02_T1_MTL.txt"
            }
        },
        "lineage": {},
    }


def expected_le07_l2_c2_folder():
    return {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": "2184a390-3bfc-5393-91fa-e9dae7c3fe39",
        "label": "usgs_ls7e_level2_2-0-20210426_090084_2021-03-31",
        "product": {
            "name": "usgs_ls7e_level2_2",
            "href": "https://collections.dea.ga.gov.au/product/usgs_ls7e_level2_2",
        },
        "properties": {
            "datetime": datetime(2021, 3, 31, 23, 1, 59, 738020),
            # The minor version comes from the processing date,
            # as used in filenames to distinguish reprocesses.
            "odc:dataset_version": "2.0.20210426",
            "odc:file_format": "GeoTIFF",
            "odc:processing_datetime": (datetime(2021, 4, 26, 10, 52, 29)),
            "odc:producer": "usgs.gov",
            "odc:product_family": "level2",
            "odc:region_code": "090084",
            "eo:cloud_cover": 6.0,
            "eo:gsd": 30.0,
            "eo:instrument": "ETM",
            "eo:platform": "landsat-7",
            "eo:sun_azimuth": 57.028_331_7,
            "eo:sun_elevation": 31.970_873_9,
            "landsat:algorithm_source_surface_reflectance": "LEDAPS_3.4.0",
            "landsat:collection_category": "T1",
            "landsat:collection_number": 2,
            "landsat:data_type": "L2SP",
            "landsat:geometric_rmse_model_x": 3.08,
            "landsat:geometric_rmse_model_y": 3.663,
            "landsat:ground_control_points_model": 1240,
            "landsat:ground_control_points_version": 5,
            "landsat:landsat_product_id": "LE07_L2SP_090084_20210331_20210426_02_T1",
            "landsat:landsat_scene_id": "LE70900842021090ASA00",
            "landsat:processing_software_version": "LPGS_15.4.0",
            "landsat:station_id": "ASA",
            "landsat:wrs_path": 90,
            "landsat:wrs_row": 84,
        },
        "crs": "epsg:32655",
        "geometry": {
            "coordinates": [
                [
                    [691984.3611711152, -3725685.0],
                    [681093.954439248, -3750395.5802668664],
                    [643185.0, -3904314.5],
                    [671569.7291070349, -3915142.15448428],
                    [825623.838894661, -3944415.0],
                    [846655.2670137166, -3944415.0],
                    [857285.0835718605, -3923426.16362107],
                    [895215.0, -3765785.5],
                    [870923.2163880672, -3754935.8100720993],
                    [713242.5838228005, -3725685.0],
                    [691984.3611711152, -3725685.0],
                ],
            ],
            "type": "Polygon",
        },
        "grids": {
            "default": {
                "shape": (60, 60),
                "transform": (
                    4200.5,
                    0.0,
                    643185.0,
                    0.0,
                    -3645.5,
                    -3725685.0,
                    0.0,
                    0.0,
                    1.0,
                ),
            },
        },
        "measurements": {
            "atmos_opacity": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_ATMOS_OPACITY.TIF"
            },
            "atmos_transmittance": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_ATRAN.TIF"
            },
            "blue": {"path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_B1.TIF"},
            "cloud_distance": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_CDIST.TIF"
            },
            "downwell_radiance": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_DRAD.TIF"
            },
            "emissivity": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_EMIS.TIF"
            },
            "emissivity_stdev": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_EMSD.TIF"
            },
            "green": {"path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_B2.TIF"},
            "lwir": {"path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_B6.TIF"},
            "nir": {"path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_B4.TIF"},
            "qa_cloud": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_CLOUD_QA.TIF"
            },
            "qa_radsat": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_QA_RADSAT.TIF"
            },
            "qa_temperature": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_QA.TIF"
            },
            "quality": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_QA_PIXEL.TIF"
            },
            "red": {"path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_B3.TIF"},
            "swir_1": {"path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_B5.TIF"},
            "swir_2": {"path": "LE07_L2SP_090084_20210331_20210426_02_T1_SR_B7.TIF"},
            "thermal_radiance": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_TRAD.TIF"
            },
            "upwell_radiance": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_ST_URAD.TIF"
            },
        },
        "accessories": {
            "metadata:landsat_mtl": {
                "path": "LE07_L2SP_090084_20210331_20210426_02_T1_MTL.txt"
            }
        },
        "lineage": {},
    }


def expected_le07_l1_c2_folder():
    return {
        "$schema": "https://schemas.opendatacube.org/dataset",
        "id": "0c2c8288-90ca-549b-8e62-5a7549418bc2",
        "label": "usgs_ls7e_level1_2-0-20220405_107068_2022-03-10",
        "product": {
            "name": "usgs_ls7e_level1_2",
            "href": "https://collections.dea.ga.gov.au/product/usgs_ls7e_level1_2",
        },
        "crs": "epsg:32652",
        "geometry": {
            "coordinates": [
                [
                    [399758.2322122816, -1256249.4492277203],
                    [399802.7308200066, -1256137.087806672],
                    [399821.55649907107, -1255918.0786010358],
                    [432005.4174861629, -1174785.0],
                    [501633.9384745485, -1174785.0],
                    [501681.19315531215, -1174792.5],
                    [505159.0, -1174792.5],
                    [643207.5, -1195584.0],
                    [643203.1032408631, -1197254.07012362],
                    [643215.0, -1197255.9583110353],
                    [643215.0, -1298829.9329701741],
                    [642978.4435009289, -1301581.9213989642],
                    [610794.582513837, -1382715.0],
                    [541166.0615254515, -1382715.0],
                    [541118.8068446879, -1382707.5],
                    [538233.3824915347, -1382707.5],
                    [399592.5, -1361646.0266450155],
                    [399592.5, -1360245.2320472337],
                    [399585.0, -1360244.0416889647],
                    [399585.0, -1258670.0670298259],
                    [399700.2956065774, -1257328.7714981567],
                    [399758.2322122816, -1256249.4492277203],
                ],
            ],
            "type": "Polygon",
        },
        "grids": {
            "default": {
                "shape": [20, 20],
                "transform": [
                    12181.5,
                    0.0,
                    399585.0,
                    0.0,
                    -10396.5,
                    -1174785.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
            "panchromatic": {
                "shape": [20, 20],
                "transform": [
                    12180.75,
                    0.0,
                    399592.5,
                    0.0,
                    -10395.75,
                    -1174792.5,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        },
        "properties": {
            "datetime": "2022-03-10T00:09:40.814477",
            "eo:cloud_cover": 5.0,
            "eo:gsd": 15.0,
            "eo:instrument": "ETM",
            "eo:platform": "landsat-7",
            "eo:sun_azimuth": 85.98764472,
            "eo:sun_elevation": 39.0330312,
            "landsat:collection_category": "T1",
            "landsat:collection_number": 2,
            "landsat:data_type": "L1TP",
            "landsat:geometric_rmse_model_x": 8.516,
            "landsat:geometric_rmse_model_y": 6.445,
            "landsat:ground_control_points_model": 6,
            "landsat:ground_control_points_version": 5,
            "landsat:landsat_product_id": "LE07_L1TP_107068_20220310_20220405_02_T1",
            "landsat:landsat_scene_id": "LE71070682022069ASA00",
            "landsat:processing_software_version": "LPGS_15.6.0",
            "landsat:station_id": "ASA",
            "landsat:wrs_path": 107,
            "landsat:wrs_row": 68,
            "odc:dataset_version": "2.0.20220405",
            "odc:file_format": "GeoTIFF",
            "odc:processing_datetime": "2022-04-05T10:37:54",
            "odc:producer": "usgs.gov",
            "odc:product_family": "level1",
            "odc:region_code": "107068",
        },
        "measurements": {
            "blue": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B1.TIF"},
            "green": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B2.TIF"},
            "nir": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B4.TIF"},
            "panchromatic": {
                "path": "LE07_L1TP_107068_20220310_20220405_02_T1_B8.TIF",
                "grid": "panchromatic",
            },
            "qa_radsat": {
                "path": "LE07_L1TP_107068_20220310_20220405_02_T1_QA_RADSAT.TIF"
            },
            "quality": {
                "path": "LE07_L1TP_107068_20220310_20220405_02_T1_QA_PIXEL.TIF"
            },
            "red": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B3.TIF"},
            "solar_azimuth": {
                "path": "LE07_L1TP_107068_20220310_20220405_02_T1_SAA.TIF"
            },
            "solar_zenith": {
                "path": "LE07_L1TP_107068_20220310_20220405_02_T1_SZA.TIF"
            },
            "swir_1": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B5.TIF"},
            "swir_2": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B7.TIF"},
            "tir_1": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B6_VCID_1.TIF"},
            "tir_2": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_B6_VCID_2.TIF"},
            "view_azimuth": {
                "path": "LE07_L1TP_107068_20220310_20220405_02_T1_VAA.TIF"
            },
            "view_zenith": {"path": "LE07_L1TP_107068_20220310_20220405_02_T1_VZA.TIF"},
        },
        "accessories": {
            "metadata:landsat_mtl": {
                "path": "LE07_L1TP_107068_20220310_20220405_02_T1_MTL.txt"
            },
        },
        "lineage": {},
    }
