import operator
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pytest
from ruamel import yaml

from eodatasets3 import DatasetAssembler, DatasetDoc, namer
from tests.common import assert_expected_eo3_path


def assert_names_match(
    tmp_path: Path,
    # Given:
    conventions,
    properties: Mapping,
    # Then expect:
    expect_metadata_path: str = None,
    expect_label: str = None,
):
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
    """
    Easily test a set of naming conventions: Do certain properties lead to expected file names?
    """

    with DatasetAssembler(tmp_path, naming_conventions=conventions) as p:
        p.properties.update(properties)

        dataset_id, metadata_path = p.done()

    if expect_metadata_path:
        metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
        assert metadata_path_offset == expect_metadata_path

    with metadata_path.open("r") as f:
        doc = yaml.YAML(typ="safe").load(f)

    if expect_label:
        assert doc["label"] == expect_label, "Unexpected dataset label"


def test_minimal_s1_dataset(tmp_path: Path):
    assert_names_match(
        tmp_path,
        conventions="default",
        properties={
            "eo:platform": "sentinel-1a",
            "eo:instrument": "c-sar",
            "datetime": datetime(2018, 11, 4),
            "odc:product_family": "bck",
            "odc:processing_datetime": "2018-11-05T12:23:23",
        },
        expect_label="s1ac_bck_2018-11-04",
        expect_metadata_path="s1ac_bck/2018/11/04/s1ac_bck_2018-11-04.odc-metadata.yaml",
    )


def test_dea_s2_derivate_names(tmp_path: Path):
    assert_names_match(
        tmp_path,
        conventions="dea_s2_derivative",
        properties={
            "eo:platform": "sentinel-2a",
            "datetime": datetime(2018, 11, 4, 5, 23, 3),
            "odc:product_family": "eucalyptus",
            "odc:processing_datetime": "2018-11-05T12:23:23",
            "odc:collection_number": 3,
            "dea:dataset_maturity": "final",
            "odc:dataset_version": "1.2.3",
            "odc:producer": "esa.int",
            "odc:region_code": "55HFA",
            "sentinel:sentinel_tile_id": "S2B_OPER_MSI_L1C_TL_EPAE_20201011T011446_A018789_T55HFA_N02.09",
        },
        expect_label="esa_s2_eucalyptus_3_55HFA_2018-11-04_final",
        expect_metadata_path="esa_s2_eucalyptus_3/1-2-3/55/HFA/2018/11/04/20201011T011446/s2a/"
        "esa_s2_eucalyptus_3_55HFA_2018-11-04_final.odc-metadata.yaml",
    )


def test_minimal_provisional_dea_dataset(tmp_path: Path):
    assert_names_match(
        tmp_path,
        conventions="dea",
        properties={
            "eo:platform": "landsat-8",
            "eo:instrument": "OLI_TIRS",
            "datetime": datetime(2020, 5, 26),
            "odc:product_family": "ufo-observations",
            "odc:processing_datetime": "2018-11-05T12:23:23",
            "odc:dataset_version": "1.0.0",
            "odc:producer": "ga.gov.au",
            "odc:region_code": "088080",
            "landsat:landsat_scene_id": "LC80880802020146LGN00",
            # Provisional! It'll be added to the path, right?
            "dea:product_maturity": "provisional",
        },
        expect_label="ga_ls8c_ufo_observations_provisional_1-0-0_088080_2020-05-26",
        expect_metadata_path="ga_ls8c_ufo_observations_provisional_1/088/080/2020/05/26/"
        "ga_ls8c_ufo_observations_provisional_1-0-0_088080_2020-05-26.odc-metadata.yaml",
    )


def test_minimal_s2_dataset_normal(tmp_path: Path):
    """A minimal dataset with sentinel platform/instrument"""
    with DatasetAssembler(tmp_path) as p:
        p.platform = "sentinel-2a"
        p.instrument = "msi"
        p.datetime = datetime(2018, 11, 4)
        p.product_family = "blueberries"
        p.processed = "2018-11-05T12:23:23"
        p.properties["sentinel:sentinel_tile_id"] = (
            "S2A_OPER_MSI_L1C_TL_SGS__20170822T015626_A011310_T54KYU_N02.05"
        )

        dataset_id, metadata_path = p.done()

    with metadata_path.open("r") as f:
        doc = yaml.YAML(typ="safe").load(f)

    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
    assert metadata_path_offset == (
        "s2am_blueberries/2018/11/04/s2am_blueberries_2018-11-04.odc-metadata.yaml"
    )

    assert doc["label"] == "s2am_blueberries_2018-11-04", "Unexpected dataset label"


def test_s2_naming_conventions(tmp_path: Path):
    """A minimal dataset with sentinel platform/instrument"""
    p = DatasetAssembler(tmp_path, naming_conventions="dea_s2")
    p.platform = "sentinel-2a"
    p.instrument = "msi"
    p.datetime = datetime(2018, 11, 4)
    p.product_family = "blueberries"
    p.processed = "2018-11-05T12:23:23"
    p.producer = "ga.gov.au"
    p.dataset_version = "1.0.0"
    p.region_code = "Oz"
    p.properties["odc:file_format"] = "GeoTIFF"
    p.properties["sentinel:sentinel_tile_id"] = (
        "S2A_OPER_MSI_L1C_TL_SGS__20170822T015626_A011310_T54KYU_N02.05"
    )

    p.note_source_datasets(
        "telemetry",
        # Accepts multiple, and they can be strings or UUIDs:
        "ca705033-0fc4-4f38-a47e-f425dfb4d0c7",
        uuid.UUID("3781e90f-b677-40af-9439-b40f6e4dfadd"),
    )

    # The property normaliser should have extracted inner fields
    assert p.properties["sentinel:datatake_start_datetime"] == datetime(
        2017, 8, 22, 1, 56, 26, tzinfo=timezone.utc
    )

    dataset_id, metadata_path = p.done()

    # The s2 naming conventions have an extra subfolder of the datatake start time.
    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()

    assert metadata_path_offset == (
        "ga_s2am_blueberries_1/Oz/2018/11/04/20170822T015626/"
        "ga_s2am_blueberries_1-0-0_Oz_2018-11-04.odc-metadata.yaml"
    )

    assert_expected_eo3_path(
        {
            "$schema": "https://schemas.opendatacube.org/dataset",
            "accessories": {},
            "id": dataset_id,
            "label": "ga_s2am_blueberries_1-0-0_Oz_2018-11-04",
            "product": {
                "href": "https://collections.dea.ga.gov.au/product/ga_s2am_blueberries_1",
                "name": "ga_s2am_blueberries_1",
            },
            "properties": {
                "datetime": datetime(2018, 11, 4, 0, 0),
                "eo:instrument": "msi",
                "eo:platform": "sentinel-2a",
                "odc:dataset_version": "1.0.0",
                "odc:file_format": "GeoTIFF",
                "odc:processing_datetime": datetime(2018, 11, 5, 12, 23, 23),
                "odc:producer": "ga.gov.au",
                "odc:product_family": "blueberries",
                "odc:region_code": "Oz",
                "sentinel:datatake_start_datetime": datetime(2017, 8, 22, 1, 56, 26),
                "sentinel:sentinel_tile_id": "S2A_OPER_MSI_L1C_TL_SGS__20170822T015626_A011310_T54KYU_N02.05",
            },
            "lineage": {
                "telemetry": [
                    "ca705033-0fc4-4f38-a47e-f425dfb4d0c7",
                    "3781e90f-b677-40af-9439-b40f6e4dfadd",
                ]
            },
        },
        expected_path=metadata_path,
    )


def test_complain_about_missing_fields(tmp_path: Path, l1_ls8_folder: Path):
    """
    It should complain immediately if I add a file without enough metadata to write the filename.

    (and with a friendly error message)
    """

    out = tmp_path / "out"
    out.mkdir()

    [blue_geotiff_path] = l1_ls8_folder.rglob("L*_B2.TIF")

    # Default simple naming conventions need at least a date and family...
    with pytest.raises(
        ValueError, match="Need more properties to fulfill naming conventions."
    ):
        with DatasetAssembler(out) as p:
            p.write_measurement("blue", blue_geotiff_path)

    # It should mention the field that's missing (we added a date, so product_family is needed)
    with DatasetAssembler(out) as p:
        with pytest.raises(ValueError, match="odc:product_family"):
            p.datetime = datetime(2019, 7, 4, 13, 7, 5)
            p.write_measurement("blue", blue_geotiff_path)

    # DEA naming conventions should have stricter standards, and will tell your which fields you need to add.
    with DatasetAssembler(out, naming_conventions="dea") as p:
        # We set all the fields that work in default naming conventions.
        p.datetime = datetime(2019, 7, 4, 13, 7, 5)
        p.product_family = "quaternarius"
        p.processed_now()

        # These fields are mandatory for DEA, and so should be complained about.
        expected_extra_fields_needed = (
            "eo:platform",
            "eo:instrument",
            "odc:dataset_version",
            "odc:producer",
            "odc:region_code",
        )
        with pytest.raises(ValueError) as got_error:
            p.write_measurement("blue", blue_geotiff_path)

        # All needed fields should have been in the error message.
        for needed_field_name in expected_extra_fields_needed:
            assert needed_field_name in got_error.value.args[0], (
                f"Expected field {needed_field_name} to "
                f"be listed as mandatory in the error message"
            )


def test_dea_interim_folder_calculation(tmp_path: Path):
    """
    DEA Naming conventions should include maturity in the folder name
    when it's not a 'final' dataset.
    """
    with DatasetAssembler(tmp_path, naming_conventions="dea") as p:
        p.platform = "landsat-7"
        # Should not end up in the path, as it's the default:
        p.product_maturity = "stable"
        p.instrument = "ETM+"
        p.datetime = datetime(1998, 7, 30)
        p.product_family = "frogs"
        p.processed = "1999-11-20 00:00:53.152462Z"
        p.maturity = "interim"
        p.producer = "ga.gov.au"
        p.properties["landsat:landsat_scene_id"] = "LE70930821999324EDC00"
        p.dataset_version = "1.2.3"
        p.region_code = "093082"

        p.done()

    [metadata_path] = tmp_path.rglob("*.odc-metadata.yaml")
    calculated_path: Path = metadata_path.relative_to(tmp_path)
    assert calculated_path == Path(
        #                                  ⇩⇩⇩⇩⇩⇩⇩⇩ Adds interim flag
        "ga_ls7e_frogs_1/093/082/1998/07/30_interim/ga_ls7e_frogs_1-2-3_093082_1998-07-30_interim.odc-metadata.yaml"
    )


def test_dea_c3_naming_conventions(tmp_path: Path):
    """
    A sample scene for Alchemist C3 processing that tests the naming conventions.
    """
    p = DatasetAssembler(tmp_path, naming_conventions="dea_c3")
    p.platform = "landsat-7"
    p.datetime = datetime(1998, 7, 30)
    p.product_family = "wo"
    p.processed = "1998-07-30T12:23:23"
    p.maturity = "interim"
    p.producer = "ga.gov.au"
    p.region_code = "090081"

    # Try missing few fields and expect ValueError
    with pytest.raises(
        ValueError, match="Need more properties to fulfill naming conventions."
    ):
        p.done()

    # Put back the missed ones
    p.dataset_version = "1.6.0"
    p.collection_number = "3"

    # Collection number returned as integer via the getter.
    assert p.collection_number == 3

    # Success case
    dataset_id, metadata_path = p.done()
    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
    assert (
        metadata_path_offset
        == "ga_ls_wo_3/1-6-0/090/081/1998/07/30/ls7/ga_ls_wo_3_090081_1998-07-30_interim.odc-metadata.yaml"
    )


def test_dataset_multi_platform(tmp_path: Path):
    """Can we make a dataset derived from multiple platforms?"""

    # No platform is included in names when there's a mix.
    with DatasetAssembler(tmp_path) as p:
        p.platforms = ["Sentinel_2a", "landsat_7"]
        assert p.platform == "landsat-7,sentinel-2a"

        p.datetime = datetime(2019, 1, 1)
        p.product_family = "peanuts"
        p.processed_now()

        dataset_id, metadata_path = p.done()

    with metadata_path.open("r") as f:
        doc = yaml.YAML(typ="safe").load(f)

    assert doc["label"] == "peanuts_2019-01-01"
    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
    assert (
        metadata_path_offset
        == "peanuts/2019/01/01/peanuts_2019-01-01.odc-metadata.yaml"
    )

    # ... but show the platform abbreviation when there's a known group.
    with DatasetAssembler(tmp_path) as p:
        p.platforms = ["Sentinel_2a", "sentinel_2b"]
        assert p.platform == "sentinel-2a,sentinel-2b"

        p.datetime = datetime(2019, 1, 1)
        p.product_family = "peanuts"
        p.processed_now()

        dataset_id, metadata_path = p.done()

    with metadata_path.open("r") as f:
        doc = yaml.YAML(typ="safe").load(f)

    assert doc["label"] == "s2_peanuts_2019-01-01"
    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
    assert (
        metadata_path_offset
        == "s2_peanuts/2019/01/01/s2_peanuts_2019-01-01.odc-metadata.yaml"
    )


def test_africa_naming_conventions(tmp_path: Path):
    """
    Minimal fields needed for DEAfrica naming conventions
    """
    with DatasetAssembler(tmp_path, naming_conventions="deafrica") as p:
        # Just the fields listed in required_fields.
        p.producer = "digitalearthafrica.org"
        p.datetime = datetime(1998, 7, 30)
        p.region_code = "090081"
        p.product_family = "wofs"
        p.platform = "LANDSAT_8"
        p.processed_now()
        p.dataset_version = "0.1.2"

        dataset_id, metadata_path = p.done()

    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
    assert (
        metadata_path_offset
        == "wofs_ls/0-1-2/090/081/1998/07/30/wofs_ls_090081_1998-07-30.odc-metadata.yaml"
    )

    with DatasetAssembler(tmp_path, naming_conventions="deafrica") as p:
        # Just the fields listed in required_fields.
        p.producer = "digitalearthafrica.org"
        p.datetime = datetime(1998, 7, 30)
        p.region_code = "090081"
        p.product_family = "fc"
        p.platform = "LANDSAT_8"
        p.processed_now()
        p.dataset_version = "0.1.2"

        dataset_id, metadata_path = p.done()

    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
    assert (
        metadata_path_offset
        == "fc_ls/0-1-2/090/081/1998/07/30/fc_ls_090081_1998-07-30.odc-metadata.yaml"
    )


def test_names_alone(tmp_path: Path):
    p = _basic_properties_set()
    convention = namer(p, conventions="dea", collection_prefix="s3://test-bucket")

    assert convention.product_name == "ga_s2am_tester_1"
    assert convention.dataset_folder == "ga_s2am_tester_1/023/543/2013/02/03"
    assert (
        convention.metadata_filename(kind="sidecar")
        == "ga_s2am_tester_1-2-3_023543_2013-02-03_sidecar.yaml"
    )

    assert convention.dataset_location == (
        "s3://test-bucket/ga_s2am_tester_1/023/543/2013/02/03/"
    )

    # Can we override generated names?

    convention.time_folder = "years/2013"
    assert convention.dataset_location == (
        "s3://test-bucket/ga_s2am_tester_1/023/543/years/2013/"
    )
    convention.region_folder = "x023y543"
    assert convention.dataset_location == (
        "s3://test-bucket/ga_s2am_tester_1/x023y543/years/2013/"
    )

    convention.dataset_folder = Path("custom/dataset/offset/")
    # Now the generated metadata path will be inside it:
    assert convention.dataset_location == "s3://test-bucket/custom/dataset/offset/"

    # Custom product name?
    convention.product_name = "my_custom_product"
    assert (
        convention.metadata_file
        == "my_custom_product-2-3_023543_2013-02-03.odc-metadata.yaml"
    )


def test_local_path_naming(tmp_path: Path):
    p = _basic_properties_set()
    # The collection prefix can be given as a local path:

    convention = namer(p, conventions="dea", collection_prefix=Path("/my/collections"))
    assert convention.dataset_location == (
        "file:///my/collections/ga_s2am_tester_1/023/543/2013/02/03/"
    )
    assert convention.resolve_file(convention.metadata_file)

    # We can get it as a pathlib object
    assert convention.dataset_path == Path(
        "/my/collections/ga_s2am_tester_1/023/543/2013/02/03"
    )


def _basic_properties_set() -> DatasetDoc:
    p = DatasetDoc()
    p.platform = "sentinel-2a"
    p.instrument = "MSI"
    p.datetime = datetime(2013, 2, 3, 6, 5, 2)
    p.region_code = "023543"
    p.processed_now()
    p.producer = "ga.gov.au"
    p.dataset_version = "1.2.3"
    p.product_family = "tester"
    return p


def test_custom_naming(tmp_path: Path):
    """
    We can create naming conventions separately, and later give it to assembler.
    """
    p = _basic_properties_set()
    convention = namer(properties=p)
    convention.dataset_folder = Path("my/custom/folder/")

    with DatasetAssembler(tmp_path, names=convention) as a:
        dataset_id, metadata_path = a.done()

    metadata_path_offset = metadata_path.relative_to(tmp_path).as_posix()
    assert (
        metadata_path_offset
        == "my/custom/folder/ga_s2am_tester_1-2-3_023543_2013-02-03.odc-metadata.yaml"
    )
