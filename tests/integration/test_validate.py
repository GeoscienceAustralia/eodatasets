import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import rasterio
from click.testing import CliRunner, Result
from rasterio.io import DatasetWriter

from eodatasets3 import serialise, validate
from eodatasets3.model import DatasetDoc

# Either a dict or a path to a document
from eodatasets3.validate import DocKind, filename_doc_kind, guess_kind_from_contents

Doc = dict | Path


@pytest.fixture()
def product():
    return dict(
        name="simple_test_product",
        description="Our test product",
        metadata={},
        license="CC-BY-SA-4.0",
        metadata_type="eo3",
        measurements=[dict(name="blue", units="1", dtype="uint8", nodata=255)],
        default_allowances=dict(
            # Allow anything used in test data. We don't need to test this warning except when done explicitly.
            allow_extra_measurements=[
                "cirrus",
                "coastal_aerosol",
                "blue",
                "green",
                "lwir_1",
                "lwir_2",
                "nir",
                "panchromatic",
                "quality",
                "red",
                "swir_1",
                "swir_2",
            ],
        ),
    )


@pytest.fixture()
def metadata_type():
    return {
        "name": "eo3",
        "description": "Minimal EO3-like",
        "dataset": {
            "id": ["id"],
            "sources": ["lineage", "source_datasets"],
            "grid_spatial": ["grid_spatial", "projection"],
            "measurements": ["measurements"],
            "creation_dt": ["properties", "odc:processing_datetime"],
            "label": ["label"],
            "format": ["properties", "odc:file_format"],
            "search_fields": {
                "time": {
                    "description": "Acquisition time range",
                    "type": "datetime-range",
                    "min_offset": [
                        ["properties", "dtr:start_datetime"],
                        ["properties", "datetime"],
                    ],
                    "max_offset": [
                        ["properties", "dtr:end_datetime"],
                        ["properties", "datetime"],
                    ],
                }
            },
        },
    }


@pytest.fixture()
def l1_ls8_product():
    """An example valid product definition, suiting the l1_ls8_dataset fixture."""
    return {
        "name": "usgs_ls8c_level1_1",
        "description": "United States Geological Survey Landsat 8 "
        "Operational Land Imager and Thermal Infra-Red Scanner Level 1 Collection 1",
        "metadata_type": "eo3_landsat_l1",
        "license": "CC-BY-4.0",
        "metadata": {
            "product": {"name": "usgs_ls8c_level1_1"},
            "properties": {
                "eo:platform": "landsat-8",
                "eo:instrument": "OLI_TIRS",
                "odc:product_family": "level1",
                "odc:producer": "usgs.gov",
                "landsat:collection_number": 1,
            },
        },
        "measurements": [
            {
                "name": "coastal_aerosol",
                "aliases": ["band01"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "blue",
                "aliases": ["band02"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "green",
                "aliases": ["band03"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "red",
                "aliases": ["band04"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "nir",
                "aliases": ["band05"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "swir_1",
                "aliases": ["band06"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "swir_2",
                "aliases": ["band07"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "panchromatic",
                "aliases": ["band08"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "cirrus",
                "aliases": ["band09"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "lwir_1",
                "aliases": ["band10"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "lwir_2",
                "aliases": ["band11"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
            {
                "name": "quality",
                "aliases": ["bqa"],
                "dtype": "uint16",
                "nodata": 65535,
                "units": "1",
            },
        ],
    }


class ValidateRunner:
    """
    Run the eo3 validator command-line interface and assert the results.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.quiet = False
        self.warnings_are_errors = False
        self.record_informational_messages = False
        self.ignore_message_codes = ["missing_suffix"]
        self.thorough: bool = False

        self.result: Result | None = None

    def assert_valid(self, *docs: Doc, expect_no_messages=True, suffix=None):
        __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
        self.run_validate(docs, suffix=suffix or ".yaml")
        was_successful = self.result.exit_code == 0
        assert was_successful, (
            f"Expected validation to succeed. Output:\n{self.result.output}"
        )

        if expect_no_messages and self.messages:
            raise AssertionError(
                "Expected no messages. Got: "
                + "\n".join(f"{k}: {v}" for k, v in self.messages.items())
            )

    def assert_invalid(self, *docs: Doc, codes: Sequence[str] = None, suffix=".yaml"):
        __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
        self.run_validate(docs, suffix=suffix)
        assert self.result.exit_code != 0, (
            f"Expected validation to fail.\n{self.result.output}"
        )

        if codes is not None:
            assert sorted(codes) == sorted(self.messages.keys()), (
                f"{sorted(codes)} != {sorted(self.messages.keys())}. Messages: {self.messages}"
            )
        else:
            assert self.result.exit_code == 1, (
                f"Expected error code 1 for 1 invalid path. Got {sorted(self.messages.items())}"
            )

    def run_validate(self, docs: Sequence[Doc], suffix=".yaml"):
        __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)

        args = ("-f", "plain")

        if self.quiet:
            args += ("-q",)
        if self.warnings_are_errors:
            args += ("-W",)
        if self.thorough:
            args += ("--thorough",)

        for i, doc in enumerate(docs):
            if isinstance(doc, Mapping):
                md_path = self.tmp_path / f"doc-{i}{suffix}"
                serialise.dump_yaml(md_path, doc)
                doc = md_path
            args += (doc,)

        self.result = CliRunner(mix_stderr=False).invoke(
            validate.run, [str(a) for a in args], catch_exceptions=False
        )

    @property
    def messages_with_severity(self) -> dict[tuple[str, str], str]:
        """
        Get all messages produced by the validation tool with their severity.

        This issimilar to ".messages", but includes all messages (including informational),
        and the user can filter by severity themselves.

        Returned as a dict of (severity, error_code) -> human_message.
        """

        def _read_message(line: str):
            severity, code, *message = line.split()
            if code in self.ignore_message_codes:
                return None, None
            return (severity, code), " ".join(message)

        messages = dict(
            _read_message(line)
            for line in self.result.stdout.split("\n")
            # message codes start with exactly one tab....
            if line and line.startswith("\t") and not line.startswith("\t\t")
        )
        # Ignored messages have key none.
        messages.pop(None, None)
        return messages

    @property
    def messages(self) -> dict[str, str]:
        """Read the messages/warnings for validation tool stdout.

        Returned as a dict of error_code -> human_message.

        (Note: this will swallow duplicates when the same error code is output multiple times.)
        """
        return {
            code: message
            for (severity, code), message in self.messages_with_severity.items()
            if (self.record_informational_messages or not severity == "I")
        }


def test_valid_document_works(eo_validator: ValidateRunner, example_metadata: dict):
    """All of our example metadata files should validate"""
    eo_validator.assert_valid(example_metadata)


def test_multi_document_works(
    tmp_path: Path,
    eo_validator: ValidateRunner,
    l1_ls5_tarball_md_expected: dict,
    l1_ls7_tarball_md_expected: dict,
):
    """We should support multiple documents in one yaml file, and validate all of them"""

    # Two valid documents in one file, should succeed.
    md_path = tmp_path / "multi-doc.yaml"
    with md_path.open("w") as f:
        serialise.dumps_yaml(f, l1_ls5_tarball_md_expected, l1_ls7_tarball_md_expected)

    eo_validator.assert_valid(md_path)

    # When the second document is invalid, we should see a validation error.
    with md_path.open("w") as f:
        e2 = dict(l1_ls5_tarball_md_expected)
        del e2["id"]
        serialise.dumps_yaml(f, l1_ls7_tarball_md_expected, e2)
    eo_validator.assert_invalid(md_path)


def test_missing_field(eo_validator: ValidateRunner, example_metadata: dict):
    """when a required field (id) is missing, validation should fail"""
    del example_metadata["id"]
    eo_validator.assert_invalid(example_metadata, codes=["structure"])
    assert "'id' is a required property" in eo_validator.messages["structure"]


def test_invalid_ls8_schema(eo_validator: ValidateRunner, example_metadata: dict):
    """When there's no eo3 $schema defined"""
    del example_metadata["$schema"]
    eo_validator.assert_invalid(
        example_metadata, codes=("no_schema",), suffix=".odc-metadata.yaml"
    )


def test_allow_optional_geo(eo_validator: ValidateRunner, example_metadata: dict):
    """A doc can omit all geo fields and be valid."""
    del example_metadata["crs"]
    del example_metadata["geometry"]

    for m in example_metadata["measurements"].values():
        if "grid" in m:
            del m["grid"]

    example_metadata["grids"] = {}
    eo_validator.assert_valid(example_metadata)

    del example_metadata["grids"]
    eo_validator.assert_valid(example_metadata)


def test_missing_geo_fields(eo_validator: ValidateRunner, example_metadata: dict):
    """If you have one gis field, you should have all of them"""
    del example_metadata["crs"]
    eo_validator.assert_invalid(example_metadata, codes=["incomplete_crs"])


def test_warn_bad_formatting(eo_validator: ValidateRunner, example_metadata: dict):
    """A warning if fields aren't formatted in standard manner."""
    example_metadata["properties"]["eo:platform"] = example_metadata["properties"][
        "eo:platform"
    ].upper()
    eo_validator.warnings_are_errors = True
    eo_validator.assert_invalid(example_metadata, codes=["property_formatting"])


def test_missing_grid_def(eo_validator: ValidateRunner, example_metadata: dict):
    """A Measurement refers to a grid that doesn't exist"""
    a_measurement, *_ = list(example_metadata["measurements"])
    example_metadata["measurements"][a_measurement]["grid"] = "unknown_grid"
    eo_validator.assert_invalid(example_metadata, codes=["invalid_grid_ref"])


def test_invalid_shape(eo_validator: ValidateRunner, example_metadata: dict):
    """the geometry must be a valid shape"""

    # Points are in an invalid order.
    example_metadata["geometry"] = {
        "coordinates": (
            (
                (770_115.0, -2_768_985.0),
                (525_285.0, -2_981_715.0),
                (770_115.0, -2_981_715.0),
                (525_285.0, -2_768_985.0),
                (770_115.0, -2_768_985.0),
            ),
        ),
        "type": "Polygon",
    }

    eo_validator.assert_invalid(example_metadata)
    assert "not a valid shape" in eo_validator.messages["invalid_geometry"]


def test_crs_as_wkt(eo_validator: ValidateRunner, example_metadata: dict):
    """A CRS should be in epsg form if an EPSG exists, not WKT"""
    example_metadata["crs"] = dedent(
        """PROJCS["WGS 84 / UTM zone 55N",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",147],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    AUTHORITY["EPSG","32655"],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
    """
    )

    # It's valid, but a warning is produced.
    eo_validator.assert_valid(example_metadata, expect_no_messages=False)

    assert "non_epsg" in eo_validator.messages
    # Suggests an alternative
    assert "change CRS to 'epsg:32655'" in eo_validator.messages["non_epsg"]

    # .. and it should fail when warnings are treated as errors.
    eo_validator.warnings_are_errors = True
    eo_validator.assert_invalid(example_metadata)


def test_valid_with_product_doc(
    eo_validator: ValidateRunner, l1_ls8_metadata_path: Path, product: dict
):
    """When a product is specified, it will validate that the measurements match the product"""

    # Document is valid on its own.
    eo_validator.assert_valid(l1_ls8_metadata_path)

    # It contains all measurements in the product, so will be valid when not thorough.
    eo_validator.assert_valid(product, l1_ls8_metadata_path)


def test_odc_product_schema(
    eo_validator: ValidateRunner, l1_ls8_metadata_path: Path, product: dict
):
    """
    If a product fails against ODC's schema, it's an error.
    """
    # A missing field will fail the schema check from ODC.
    # (these cannot be added to ODC so are a hard validation failure)
    del product["metadata"]
    eo_validator.assert_invalid(product, codes=["document_schema"])


def test_warn_bad_product_license(
    eo_validator: ValidateRunner, l1_ls8_metadata_path: Path, product: dict
):
    # Missing license is a warning.
    del product["license"]
    eo_validator.assert_valid(product, expect_no_messages=False)
    assert eo_validator.messages_with_severity == {
        ("W", "no_license"): "Product 'simple_test_product' has no license field"
    }

    # Invalid license string (not SPDX format), error. Is caught by ODC schema.
    product["license"] = "Sorta Creative Commons"
    eo_validator.assert_invalid(product, codes=["document_schema"])


def test_warn_duplicate_measurement_name(
    eo_validator: ValidateRunner,
    l1_ls8_product: dict,
):
    """When a product is specified, it will validate that names are not repeated between measurements and aliases."""
    product = l1_ls8_product
    # We have the "blue" measurement twice.
    product["measurements"].append(
        dict(name="blue", dtype="uint8", units="1", nodata=255),
    )

    eo_validator.assert_invalid(product)
    assert eo_validator.messages == {
        "duplicate_measurement_name": "Name 'blue' is used by multiple measurements"
    }

    # An *alias* clashes with the *name* of a measurement.
    product["measurements"].append(
        dict(
            name="azul",
            aliases=[
                "icecream",
                # Clashes with the *name* of a measurement.
                "blue",
            ],
            units="1",
            dtype="uint8",
            nodata=255,
        ),
    )
    eo_validator.assert_invalid(product)
    assert eo_validator.messages == {
        "duplicate_measurement_name": "Name 'blue' is used by multiple measurements"
    }

    # An alias is duplicated on the same measurement. Not an error, just a message!
    product["measurements"] = [
        dict(
            name="blue",
            aliases=[
                "icecream",
                "blue",
            ],
            dtype="uint8",
            units="1",
            nodata=255,
        ),
    ]
    eo_validator.assert_valid(product)
    assert eo_validator.messages_with_severity == {
        (
            "I",
            "duplicate_alias_name",
        ): "Measurement 'blue' has a duplicate alias named 'blue'"
    }


def test_dtype_compare_with_product_doc(
    eo_validator: ValidateRunner, l1_ls8_metadata_path: Path, product: dict
):
    """'thorough' validation should check the dtype of measurements against the product"""

    product["measurements"] = [dict(name="blue", dtype="uint8", units="1", nodata=255)]

    # When thorough, the dtype and nodata are wrong
    eo_validator.thorough = True
    eo_validator.assert_invalid(
        product, l1_ls8_metadata_path, codes=["different_dtype"]
    )
    assert eo_validator.messages == {
        "different_dtype": "blue dtype: product 'uint8' != dataset 'uint16'"
    }


def test_nodata_compare_with_product_doc(
    eo_validator: ValidateRunner,
    l1_ls8_dataset: DatasetDoc,
    l1_ls8_metadata_path: Path,
    l1_ls8_product: dict,
):
    """'thorough' validation should check the nodata of measurements against the product"""
    eo_validator.thorough = True
    eo_validator.record_informational_messages = True

    # Remake the tiff with a 'nodata' set.
    blue_tif = l1_ls8_metadata_path.parent / l1_ls8_dataset.measurements["blue"].path
    _create_dummy_tif(
        blue_tif,
        dtype="uint16",
        nodata=65535,
    )
    eo_validator.assert_valid(
        l1_ls8_product, l1_ls8_metadata_path, expect_no_messages=True
    )

    # Override blue definition with invalid nodata value.
    _measurement(l1_ls8_product, "blue")["nodata"] = 255

    eo_validator.assert_invalid(l1_ls8_product, l1_ls8_metadata_path)
    assert eo_validator.messages == {
        "different_nodata": "blue nodata: product 255 != dataset 65535.0"
    }


def test_measurements_compare_with_nans(
    eo_validator: ValidateRunner,
    l1_ls8_dataset: DatasetDoc,
    l1_ls8_metadata_path: Path,
    l1_ls8_product: dict,
):
    """When dataset and product have NaN nodata values, it should handle them correctly"""
    product = l1_ls8_product
    eo_validator.thorough = True
    eo_validator.record_informational_messages = True
    blue_tif = l1_ls8_metadata_path.parent / l1_ls8_dataset.measurements["blue"].path

    # When both are NaN, it should be valid
    blue = _measurement(product, "blue")
    blue["nodata"] = float("NaN")
    blue["dtype"] = "float32"
    _create_dummy_tif(blue_tif, nodata=float("NaN"))
    eo_validator.assert_valid(product, l1_ls8_metadata_path, expect_no_messages=True)

    # ODC can also represent NaNs as strings due to json's lack of NaN
    blue["nodata"] = "NaN"
    eo_validator.assert_valid(product, l1_ls8_metadata_path, expect_no_messages=True)

    # When product is set, dataset is NaN, they no longer match.
    blue["nodata"] = 0
    eo_validator.assert_invalid(product, l1_ls8_metadata_path)
    assert eo_validator.messages == {
        "different_nodata": "blue nodata: product 0 != dataset nan"
    }


def _measurement(product: dict, name: str):
    """Get a measurement by name"""
    for m in product["measurements"]:
        if m["name"] == name:
            return m
    raise ValueError(f"Measurement {name} not found?")


def _create_dummy_tif(blue_tif, nodata=None, dtype="float32", **opts):
    with rasterio.open(
        blue_tif,
        "w",
        width=10,
        height=10,
        count=1,
        dtype=dtype,
        driver="GTiff",
        nodata=nodata,
        **opts,
    ) as ds:
        ds: DatasetWriter
        ds.write(np.ones((10, 10), dtype=dtype), 1)


def test_missing_measurement_from_product(
    eo_validator: ValidateRunner,
    l1_ls8_metadata_path: Path,
    product: dict,
):
    """Validator should notice a missing measurement from the product def"""
    product["name"] = "test_with_extra_measurement"
    product["measurements"] = [
        dict(name="razzmatazz", dtype="int32", units="1", nodata=-999)
    ]

    eo_validator.assert_invalid(product, l1_ls8_metadata_path)
    assert eo_validator.messages == {
        "missing_measurement": "Product test_with_extra_measurement expects a measurement 'razzmatazz')"
    }


def test_supports_measurementless_products(
    eo_validator: ValidateRunner,
    l1_ls8_metadata_path: Path,
    product: dict,
):
    """
    Validator should support products without any measurements in the document.

    These are valid for products which can't be dc.load()'ed but are
    referred to for provenance, such as DEA's telemetry data or DEA's collection-2
    Level 1 products.
    """
    product["measurements"] = []
    eo_validator.assert_valid(product, l1_ls8_metadata_path)


def test_complains_about_measurement_lists(
    eo_validator: ValidateRunner,
    l1_ls8_metadata_path: Path,
    product: dict,
):
    """
    Complain when product measurements are a dict.

    datasets have measurements as a dict, products have them as a List, so this is a common error.
    """

    product["measurements"] = {"a": {}}
    eo_validator.assert_invalid(product)
    assert (
        eo_validator.messages.get("measurements_list")
        == "Product measurements should be a list/sequence (Found a 'dict')."
    )


def test_complains_about_product_not_matching(
    eo_validator: ValidateRunner,
    l1_ls8_metadata_path: Path,
    product: dict,
):
    """
    Complains when we're given products but they don't match the dataset
    """

    # A metadata field that's not in the dataset.
    product["metadata"]["favourite_sandwich"] = "cucumber"

    eo_validator.assert_invalid(product, l1_ls8_metadata_path)
    assert (
        eo_validator.messages.get("unknown_product")
        == "Dataset does not match the given products"
    )


def test_complains_about_impossible_nodata_vals(
    eo_validator: ValidateRunner,
    l1_ls8_metadata_path: Path,
    product: dict,
):
    """Complain if a product nodata val cannot be represented in the dtype"""

    product["measurements"].append(
        dict(
            name="paradox",
            dtype="uint8",
            units="1",
            # Impossible for a uint6
            nodata=-999,
        )
    )
    eo_validator.assert_invalid(product)
    assert eo_validator.messages == {
        "unsuitable_nodata": "Measurement 'paradox' nodata -999 does not fit a 'uint8'"
    }


def test_complains_when_no_product(
    eo_validator: ValidateRunner, l1_ls8_metadata_path: Path
):
    """When a product is specified, it will validate that the measurements match the product"""
    # Thorough checking should fail when there's no product provided
    eo_validator.thorough = True
    eo_validator.record_informational_messages = True
    eo_validator.assert_invalid(l1_ls8_metadata_path, codes=["no_product"])


def test_validate_metadata_type(eo_validator: ValidateRunner, metadata_type: Doc):
    eo_validator.assert_valid(metadata_type, suffix=".odc-type.yaml")
    eo_validator.assert_valid(metadata_type)
    del metadata_type["dataset"]["id"]
    eo_validator.assert_invalid(
        metadata_type, codes=["document_schema"], suffix=".odc-type.yaml"
    )


def test_is_product():
    """Product documents should be correctly identified as products"""
    product = dict(
        name="minimal_product", metadata_type="eo3", measurements=[dict(name="blue")]
    )
    assert guess_kind_from_contents(product) == DocKind.product


def test_dataset_is_not_a_product(example_metadata: dict):
    """
    Datasets should not be identified as products

    (checks all example metadata files)
    """
    assert guess_kind_from_contents(example_metadata) == DocKind.dataset
    assert filename_doc_kind(Path("asdf.odc-metadata.yaml")) == DocKind.dataset


def test_get_field_offsets(metadata_type: dict):
    """
    Test the get_field_offsets function.
    """
    assert list(validate._get_field_offsets(metadata_type)) == [
        ("id", [["id"]]),
        ("sources", [["lineage", "source_datasets"]]),
        ("grid_spatial", [["grid_spatial", "projection"]]),
        ("measurements", [["measurements"]]),
        ("creation_dt", [["properties", "odc:processing_datetime"]]),
        ("label", [["label"]]),
        ("format", [["properties", "odc:file_format"]]),
        (
            "time",
            [
                ["properties", "dtr:start_datetime"],
                ["properties", "datetime"],
                ["properties", "dtr:end_datetime"],
                ["properties", "datetime"],
            ],
        ),
    ]


@pytest.fixture
def eo_validator(tmp_path) -> ValidateRunner:
    return ValidateRunner(tmp_path)
