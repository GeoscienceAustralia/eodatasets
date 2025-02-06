"""
Validate ODC dataset documents
"""

import collections
import enum
import math
import multiprocessing
import os
import sys
from collections import Counter
from collections.abc import Generator, Iterable, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from textwrap import indent
from typing import (
    Optional,
)
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import attr
import cattr
import ciso8601
import click
import numpy as np
import rasterio
import toolz
from attr import Factory, define, field, frozen
from boltons.iterutils import get_path
from click import echo, secho, style
from datacube import Datacube
from datacube.index.eo3 import prep_eo3
from datacube.utils import InvalidDocException, changes, is_url, read_documents
from datacube.utils.documents import load_documents
from rasterio import DatasetReader
from rasterio.crs import CRS
from rasterio.errors import CRSError
from shapely.validation import explain_validity

from eodatasets3 import model, serialise, utils
from eodatasets3.model import DatasetDoc
from eodatasets3.ui import bool_style, is_absolute, uri_resolve
from eodatasets3.utils import EO3_SCHEMA, default_utc

DEFAULT_NULLABLE_FIELDS = ("label",)
DEFAULT_OPTIONAL_FIELDS = (
    # Older product do not have this field at all, and when not specified it is considered stable.
    "dataset_maturity",
)


class Level(enum.Enum):
    info = 1
    warning = 2
    error = 3


class DocKind(enum.Enum):
    # EO3 datacube dataset.
    dataset = 1
    # Datacube product
    product = 2
    # Datacube Metadata Type
    metadata_type = 3
    # Stac Item
    stac_item = 4
    # Legacy datacube ("eo1") dataset
    legacy_dataset = 5
    # Legacy product config for ingester
    ingestion_config = 6


# What kind of document each suffix represents.
# (full suffix will also have a doc type: .yaml, .json, .yaml.gz etc)
# Example:  "my-test-dataset.odc-metadata.yaml"
SUFFIX_KINDS = {
    ".odc-metadata": DocKind.dataset,
    ".odc-product": DocKind.product,
    ".odc-type": DocKind.metadata_type,
}
# Inverse of above
DOC_TYPE_SUFFIXES = {v: k for k, v in SUFFIX_KINDS.items()}


def filename_doc_kind(path: str | Path) -> Optional["DocKind"]:
    """
    Get the expected file type for the given filename.

    Returns None if it does not follow any naming conventions.

    >>> filename_doc_kind('LC8_2014.odc-metadata.yaml').name
    'dataset'
    >>> filename_doc_kind('/tmp/something/water_bodies.odc-metadata.yaml.gz').name
    'dataset'
    >>> filename_doc_kind(Path('/tmp/something/ls8_fc.odc-product.yaml')).name
    'product'
    >>> filename_doc_kind(Path('/tmp/something/ls8_wo.odc-product.json.gz')).name
    'product'
    >>> filename_doc_kind(Path('/tmp/something/eo3_gqa.odc-type.yaml')).name
    'metadata_type'
    >>> filename_doc_kind(Path('/tmp/something/some_other_file.yaml'))
    """

    for suffix in reversed(Path(path).suffixes):
        suffix = suffix.lower()
        if suffix in SUFFIX_KINDS:
            return SUFFIX_KINDS[suffix]

    return None


def guess_kind_from_contents(doc: dict):
    """
    What sort of document do the contents look like?
    """
    if "$schema" in doc and doc["$schema"] == EO3_SCHEMA:
        return DocKind.dataset
    if "metadata_type" in doc:
        if "source_type" in doc:
            return DocKind.ingestion_config
        return DocKind.product
    if ("dataset" in doc) and ("search_fields" in doc["dataset"]):
        return DocKind.metadata_type
    if "id" in doc:
        if ("lineage" in doc) and ("platform" in doc):
            return DocKind.legacy_dataset

        if ("properties" in doc) and ("datetime" in doc["properties"]):
            return DocKind.stac_item

    return None


@frozen
class ValidationMessage:
    level: Level
    code: str
    reason: str
    hint: str = None
    #: What was assumed when validating this? Eg: dict(product='ls7_nbar', metadata_type='eo3')
    context: dict = None

    def __str__(self) -> str:
        hint = ""
        if self.hint:
            hint = f" (Hint: {self.hint})"
        return f"{self.code}: {self.reason}{hint}"


def _info(code: str, reason: str, hint: str = None):
    return ValidationMessage(Level.info, code, reason, hint=hint)


def _warning(code: str, reason: str, hint: str = None):
    return ValidationMessage(Level.warning, code, reason, hint=hint)


def _error(code: str, reason: str, hint: str = None):
    return ValidationMessage(Level.error, code, reason, hint=hint)


ValidationMessages = Generator[ValidationMessage, None, None]


@frozen(init=True)
class ValidationExpectations:
    """
    What expectations do we have when validating this dataset?
    """

    #: Allow these extra measurement names to be included in the dataset.
    #: (ODC allows unlisted measurement names, but it's usually a mistake)
    allow_extra_measurements: Sequence[str] = ()

    #: Do we expect full geometry information in every dataset?
    #: (It's optional in ODC, but often a mistake to miss it)
    require_geometry: bool = True

    #: Are any of the configured fields nullable?
    allow_nullable_fields: Sequence[str] = field(
        default=Factory(lambda: DEFAULT_NULLABLE_FIELDS)
    )
    #: Can any of the fields be completely omitted from the document?
    allow_missing_fields: Sequence[str] = field(
        default=Factory(lambda: DEFAULT_OPTIONAL_FIELDS)
    )

    def with_document_overrides(self, doc: dict):
        """
        Return an instance with any overrides from the given document.
        """
        if "default_allowances" not in doc:
            return self

        overridden_values = {**attr.asdict(self), **doc["default_allowances"]}
        # Merge, don't replace, these lists.
        overridden_values["allow_nullable_fields"] = list(
            {*overridden_values["allow_nullable_fields"], *self.allow_nullable_fields}
        )
        overridden_values["allow_missing_fields"] = list(
            {*overridden_values["allow_missing_fields"], *self.allow_missing_fields}
        )
        overridden_values["allow_extra_measurements"] = list(
            {
                *overridden_values["allow_extra_measurements"],
                *self.allow_extra_measurements,
            }
        )
        return cattr.structure(overridden_values, self.__class__)


def validate_dataset(
    doc: dict,
    product_definition: dict | None = None,
    metadata_type_definition: dict | None = None,
    thorough: bool = False,
    readable_location: str | Path = None,
    expect: ValidationExpectations = None,
) -> ValidationMessages:
    """
    Validate a dataset document, optionally against the given product.

    By default this will only look at the metadata, run with thorough=True to
    open the data files too.

    :param product_definition: Optionally check that the dataset matches this product definition.
    :param thorough: Open the imagery too, to check that data types etc match.
    :param readable_location: Dataset location to use, if not the metadata path.
    :param expect: Where can we be lenient in validation?
    """
    validation_context = {}
    expect = expect or ValidationExpectations()
    if metadata_type_definition is not None:
        expect = expect.with_document_overrides(metadata_type_definition)
        validation_context["type"] = metadata_type_definition["name"]
    if product_definition is not None:
        expect = expect.with_document_overrides(product_definition)
        validation_context["product"] = product_definition["name"]

    # noinspection PyShadowingNames
    def _info(code: str, reason: str, hint: str = None):
        return ValidationMessage(
            Level.info, code, reason, hint=hint, context=validation_context
        )

    # noinspection PyShadowingNames
    def _warning(code: str, reason: str, hint: str = None):
        return ValidationMessage(
            Level.warning, code, reason, hint=hint, context=validation_context
        )

    # noinspection PyShadowingNames
    def _error(code: str, reason: str, hint: str = None):
        return ValidationMessage(
            Level.error, code, reason, hint=hint, context=validation_context
        )

    schema = doc.get("$schema")
    if schema is None:
        yield _error(
            "no_schema",
            f"No $schema field. "
            f"You probably want an ODC dataset schema {model.ODC_DATASET_SCHEMA_URL!r}",
        )
        return
    if schema != model.ODC_DATASET_SCHEMA_URL:
        yield _error(
            "unknown_doc_type",
            f"Unknown doc schema {schema!r}. Only ODC datasets are supported ({model.ODC_DATASET_SCHEMA_URL!r})",
        )
        return

    has_doc_errors = False
    for error in serialise.DATASET_SCHEMA.iter_errors(doc):
        has_doc_errors = True
        displayable_path = ".".join(error.absolute_path)

        hint = None
        if displayable_path == "crs" and "not of type" in error.message:
            hint = "epsg codes should be prefixed with 'epsg:1234'"

        context = f"({displayable_path}) " if displayable_path else ""
        yield _error("structure", f"{context}{error.message} ", hint=hint)

    if has_doc_errors:
        return

    dataset = serialise.from_doc(doc, skip_validation=True)

    if not dataset.product.href:
        _info("product_href", "A url (href) is recommended for products")

    yield from _validate_geo(dataset, expect_geometry=expect.require_geometry)

    # Note that a dataset may have no measurements (eg. telemetry data).
    # (TODO: a stricter mode for when we know we should have geo and measurement info)
    if dataset.measurements:
        for name, measurement in dataset.measurements.items():
            grid_name = measurement.grid
            if grid_name != "default" or dataset.grids:
                if grid_name not in dataset.grids:
                    yield _error(
                        "invalid_grid_ref",
                        f"Measurement {name!r} refers to unknown grid {grid_name!r}",
                    )

            if is_absolute(measurement.path):
                yield _warning(
                    "absolute_path",
                    f"measurement {name!r} has an absolute path: {measurement.path!r}",
                )

    yield from _validate_stac_properties(dataset)

    required_measurements: dict[str, ExpectedMeasurement] = {}
    if product_definition is not None:
        required_measurements.update(
            {
                m.name: m
                for m in map(
                    ExpectedMeasurement.from_definition,
                    product_definition.get("measurements") or (),
                )
            }
        )

        product_name = product_definition.get("name")
        if product_name != dataset.product.name:
            # This is only informational as it's possible products may be indexed with finer-grained
            # categories than the original datasets: eg. a separate "nrt" product, or test product.
            yield _info(
                "product_mismatch",
                f"Dataset product name {dataset.product.name!r} "
                f"does not match the given product ({product_name!r}",
            )

        for name in required_measurements:
            if name not in dataset.measurements.keys():
                yield _error(
                    "missing_measurement",
                    f"Product {product_name} expects a measurement {name!r})",
                )
        measurements_not_in_product = set(dataset.measurements.keys()).difference(
            {m["name"] for m in product_definition.get("measurements") or ()}
        )
        # Remove the measurements that are allowed to be extra.
        measurements_not_in_product.difference_update(
            expect.allow_extra_measurements or set()
        )

        if measurements_not_in_product:
            things = ", ".join(sorted(measurements_not_in_product))
            yield _warning(
                "extra_measurements",
                f"Dataset has measurements not present in product definition for {product_name!r}: {things}",
                hint="This may be valid, as it's allowed by ODC. Set `expect_extra_measurements` to mute this.",
            )

    if metadata_type_definition:
        # Datacube does certain transforms on an eo3 doc before storage.
        # We need to do the same, as the fields will be read from the storage.
        prepared_doc = prep_eo3(doc)

        all_nullable_fields = tuple(expect.allow_nullable_fields) + tuple(
            expect.allow_missing_fields
        )
        for field_name, offsets in _get_field_offsets(
            metadata_type=metadata_type_definition
        ):
            if (
                # If a field is required...
                (field_name not in expect.allow_missing_fields)
                and
                # ... and none of its offsets are in the document
                not any(_has_offset(prepared_doc, offset) for offset in offsets)
            ):
                # ... warn them.
                product_name = (
                    product_definition.get("name")
                    if product_definition
                    else dataset.product.name
                )
                readable_offsets = " or ".join("->".join(offset) for offset in offsets)
                yield _warning(
                    "missing_field",
                    f"Dataset is missing field {field_name!r} "
                    f"for type {metadata_type_definition['name']!r}",
                    hint=f"Expected at {readable_offsets}",
                )
                continue

            if field_name not in all_nullable_fields:
                value = None
                for offset in offsets:
                    value = toolz.get_in(offset, prepared_doc)
                if value is None:
                    yield _info(
                        "null_field",
                        f"Value is null for configured field {field_name!r}",
                    )

    dataset_location = dataset.locations[0] if dataset.locations else readable_location

    # If we have a location:
    # For each measurement, try to load it.
    # If loadable:
    if thorough:
        for name, measurement in dataset.measurements.items():
            full_path = uri_resolve(dataset_location, measurement.path)
            expected_measurement = required_measurements.get(name)

            band = measurement.band or 1
            with rasterio.open(full_path) as ds:
                ds: DatasetReader

                if band not in ds.indexes:
                    yield _error(
                        "incorrect_band",
                        f"Measurement {name!r} file contains no rio index {band!r}.",
                        hint=f"contains indexes {ds.indexes!r}",
                    )
                    continue

                if not expected_measurement:
                    # The measurement is not in the product definition
                    #
                    # This is only informational because a product doesn't have to define all
                    # measurements that the datasets contain.
                    #
                    # This is historically because dataset documents reflect the measurements that
                    # are stored on disk, which can differ. But products define the set of measurments
                    # that are mandatory in every dataset.
                    #
                    # (datasets differ when, for example, sensors go offline, or when there's on-disk
                    #  measurements like panchromatic that GA doesn't want in their product definitions)
                    if required_measurements:
                        yield _info(
                            "unspecified_measurement",
                            f"Measurement {name} is not in the product",
                        )
                else:
                    expected_dtype = expected_measurement.dtype
                    band_dtype = ds.dtypes[band - 1]
                    # TODO: NaN handling
                    if expected_dtype != band_dtype:
                        yield _error(
                            "different_dtype",
                            f"{name} dtype: "
                            f"product {expected_dtype!r} != dataset {band_dtype!r}",
                        )

                    ds_nodata = ds.nodatavals[band - 1]

                    # If the dataset is missing 'nodata', we can allow anything in product 'nodata'.
                    # (In ODC, nodata might be a fill value for loading data.)
                    if ds_nodata is None:
                        continue

                    # Otherwise check that nodata matches.
                    expected_nodata = expected_measurement.nodata
                    if expected_nodata != ds_nodata and not (
                        _is_nan(expected_nodata) and _is_nan(ds_nodata)
                    ):
                        yield _error(
                            "different_nodata",
                            f"{name} nodata: "
                            f"product {expected_nodata!r} != dataset {ds_nodata!r}",
                        )


def _has_offset(doc: dict, offset: list[str]) -> bool:
    """
    Is the given offset present in the document?
    """
    for key in offset:
        if key not in doc:
            return False
        doc = doc[key]
    return True


def validate_product(doc: dict) -> ValidationMessages:
    """
    Check for common product mistakes
    """

    # Validate it against ODC's product schema.
    has_doc_errors = False
    for error in serialise.PRODUCT_SCHEMA.iter_errors(doc):
        has_doc_errors = True
        displayable_path = ".".join(map(str, error.absolute_path))
        context = f"({displayable_path}) " if displayable_path else ""
        yield _error("document_schema", f"{context}{error.message} ")

    # The jsonschema error message for this (common error) is garbage. Make it clearer.
    measurements = doc.get("measurements")
    if (measurements is not None) and not isinstance(measurements, Sequence):
        yield _error(
            "measurements_list",
            f"Product measurements should be a list/sequence "
            f"(Found a {type(measurements).__name__!r}).",
        )

    # There's no point checking further if the core doc structure is wrong.
    if has_doc_errors:
        return

    if not doc.get("license", "").strip():
        yield _warning(
            "no_license",
            f"Product {doc['name']!r} has no license field",
            hint='Eg. "CC-BY-4.0" (SPDX format), "various" or "proprietary"',
        )

    # Check measurement name clashes etc.
    if measurements is None:
        # Products don't have to have measurements. (eg. provenance-only products)
        ...
    else:
        seen_names_and_aliases = collections.defaultdict(list)
        for measurement in measurements:
            measurement_name = measurement.get("name")
            dtype = measurement.get("dtype")
            nodata = measurement.get("nodata")
            if not numpy_value_fits_dtype(nodata, dtype):
                yield _error(
                    "unsuitable_nodata",
                    f"Measurement {measurement_name!r} nodata {nodata!r} does not fit a {dtype!r}",
                )

            # Were any of the names seen in other measurements?
            these_names = measurement_name, *measurement.get("aliases", ())
            for new_field_name in these_names:
                measurements_with_this_name = seen_names_and_aliases[new_field_name]
                if measurements_with_this_name:
                    seen_in = " and ".join(
                        repr(s)
                        for s in ([measurement_name, *measurements_with_this_name])
                    )

                    # If the same name is used by different measurements, its a hard error.
                    yield _error(
                        "duplicate_measurement_name",
                        f"Name {new_field_name!r} is used by multiple measurements",
                        hint=f"It's duplicated in an alias. "
                        f"Seen in measurement(s) {seen_in}",
                    )

            # Are any names duplicated within the one measurement? (not an error, but info)
            for duplicate_name in _find_duplicates(these_names):
                yield _info(
                    "duplicate_alias_name",
                    f"Measurement {measurement_name!r} has a duplicate alias named {duplicate_name!r}",
                )

            for field_ in these_names:
                seen_names_and_aliases[field_].append(measurement_name)


def validate_metadata_type(doc: dict) -> ValidationMessages:
    """
    Check for common metadata-type mistakes
    """

    # Validate it against ODC's schema (there will be refused by ODC otherwise)
    for error in serialise.METADATA_TYPE_SCHEMA.iter_errors(doc):
        displayable_path = ".".join(map(str, error.absolute_path))
        context = f"({displayable_path}) " if displayable_path else ""
        yield _error("document_schema", f"{context}{error.message} ")


def _find_duplicates(values: Iterable[str]) -> Generator[str, None, None]:
    """Return any duplicate values in the given sequence

    >>> list(_find_duplicates(('a', 'b', 'c')))
    []
    >>> list(_find_duplicates(('a', 'b', 'b')))
    ['b']
    >>> list(_find_duplicates(('a', 'b', 'b', 'a')))
    ['a', 'b']
    """
    previous = None
    for v in sorted(values):
        if v == previous:
            yield v
        previous = v


def numpy_value_fits_dtype(value, dtype):
    """
    Can the value be exactly represented by the given numpy dtype?

    >>> numpy_value_fits_dtype(3, 'uint8')
    True
    >>> numpy_value_fits_dtype(3, np.dtype('uint8'))
    True
    >>> numpy_value_fits_dtype(-3, 'uint8')
    False
    >>> numpy_value_fits_dtype(3.5, 'float32')
    True
    >>> numpy_value_fits_dtype(3.5, 'int16')
    False
    >>> numpy_value_fits_dtype(float('NaN'), 'float32')
    True
    >>> numpy_value_fits_dtype(float('NaN'), 'int32')
    False
    """
    dtype = np.dtype(dtype)

    if value is None:
        value = 0

    if _is_nan(value):
        return np.issubdtype(dtype, np.floating)
    else:
        try:
            return bool(np.all(np.array([value], dtype=dtype) == [value]))
        except Exception:  # pylint: disable=broad-except
            return False


@define
class ExpectedMeasurement:
    name: str
    dtype: str
    nodata: int

    @classmethod
    def from_definition(cls, doc: dict):
        return ExpectedMeasurement(doc["name"], doc.get("dtype"), doc.get("nodata"))


# Name of a field and its possible offsets in the document.
FieldNameOffsetS = tuple[str, set[list[str]]]


def validate_paths(
    paths: list[str],
    thorough: bool = False,
    product_definitions: dict[str, dict] = None,
    metadata_type_definitions: dict[str, dict] = None,
    expect: ValidationExpectations = None,
) -> Generator[tuple[str, list[ValidationMessage]], None, None]:
    """Validate the list of paths. Product documents can be specified before their datasets."""

    products = dict(product_definitions or {})
    metadata_types = dict(metadata_type_definitions or {})

    for url, doc, was_specified_by_user in read_paths(paths):
        messages = []
        kind = filename_doc_kind(url)
        if kind is None:
            kind = guess_kind_from_contents(doc)
            if kind and (kind in DOC_TYPE_SUFFIXES):
                # It looks like an ODC doc but doesn't have the standard suffix.
                messages.append(
                    _warning(
                        "missing_suffix",
                        f"Document looks like a {kind.name} but does not have "
                        f'filename extension "{DOC_TYPE_SUFFIXES[kind]}{_readable_doc_extension(url)}"',
                    )
                )

        if kind == DocKind.product:
            messages.extend(validate_product(doc))
            if "name" in doc:
                products[doc["name"]] = doc
        elif kind == DocKind.dataset:
            messages.extend(
                validate_eo3_doc(
                    doc,
                    url,
                    products,
                    metadata_types,
                    thorough,
                    expect=expect,
                )
            )
        elif kind == DocKind.metadata_type:
            messages.extend(validate_metadata_type(doc))
            if "name" in doc:
                metadata_types[doc["name"]] = doc

        # Otherwise it's a file we don't support.
        # If the user gave us the path explicitly, it seems to be an error.
        # (if they didn't -- it was found via scanning directories -- we don't care.)
        elif was_specified_by_user:
            if kind is None:
                raise ValueError(f"Unknown document type for {url}")
            else:
                raise NotImplementedError(
                    f"Cannot currently validate {kind.name} files"
                )
        else:
            # Not a doc type we recognise, and the user didn't specify it. Skip it.
            continue

        yield url, messages


def _get_field_offsets(metadata_type: dict) -> Iterable[FieldNameOffsetS]:
    """
    Yield all fields and their possible document-offsets that are expected for this metadata type.

    Eg, if the metadata type has a region_code field expected properties->region_code, this
    will yield ('region_code', {['properties', 'region_code']})

    (Properties can have multiple offsets, where ODC will choose the first non-null one, hence the
    return of multiple offsets for each field.)
    """
    dataset_section = metadata_type["dataset"]
    search_fields = dataset_section["search_fields"]

    # The fixed fields of ODC. 'id', 'label', etc.
    for field_ in dataset_section:
        if field_ == "search_fields":
            continue

        offset = dataset_section[field_]
        if offset is not None:
            yield field_, [offset]

    # The configurable search fields.
    for field_, spec in search_fields.items():
        offsets = []
        if "offset" in spec:
            offsets.append(spec["offset"])
        offsets.extend(spec.get("min_offset", []))
        offsets.extend(spec.get("max_offset", []))

        yield field_, offsets


def _readable_doc_extension(uri: str):
    """
    >>> _readable_doc_extension('something.json.gz')
    '.json.gz'
    >>> _readable_doc_extension('something.yaml')
    '.yaml'
    >>> _readable_doc_extension('apple.odc-metadata.yaml.gz')
    '.yaml.gz'
    >>> _readable_doc_extension('products/tmad/tmad_product.yaml#part=1')
    '.yaml'
    >>> _readable_doc_extension('/tmp/human.06.tall.yml')
    '.yml'
    >>> # Not a doc, even though it's compressed.
    >>> _readable_doc_extension('db_dump.gz')
    >>> _readable_doc_extension('/tmp/nothing')
    """
    path = urlparse(uri).path
    compression_formats = (".gz",)
    doc_formats = (
        ".yaml",
        ".yml",
        ".json",
    )
    suffix = "".join(
        s.lower()
        for s in Path(path).suffixes
        if s.lower() in doc_formats + compression_formats
    )
    # If it's only compression, no doc format, it's not valid.
    if suffix in compression_formats:
        return None
    return suffix or None


def read_paths(
    input_paths: Iterable[str],
) -> Generator[tuple[str, dict | str, bool], None, None]:
    """
    Read the given input paths, returning a URL, document, and whether
    it was explicitly given by the user.

    When a local directory is specified, inner readable docs are returned, but will
    be marked as not explicitly specified.
    """
    for input_ in input_paths:
        for uri, was_specified in expand_paths_as_uris([input_]):
            try:
                for full_uri, doc in read_documents(uri, uri=True):
                    yield full_uri, doc, was_specified
            except InvalidDocException as e:
                if was_specified:
                    raise
                else:
                    echo(e, err=True)


def expand_paths_as_uris(
    input_paths: Iterable[str],
) -> Generator[tuple[Path, bool], None, None]:
    """
    For any paths that are directories, find inner documents that are known.

    Returns Tuples: path as a URL, and whether it was specified explicitly by user.
    """
    for input_ in input_paths:
        if is_url(input_):
            yield input_, True
        else:
            path = Path(input_).resolve()
            if path.is_dir():
                for found_path in path.rglob("*"):
                    if _readable_doc_extension(found_path.as_uri()) is not None:
                        yield found_path.as_uri(), False
            else:
                yield path.as_uri(), True


def validate_eo3_doc(
    doc: dict,
    location: str | Path,
    products: dict[str, dict],
    metadata_types: dict[str, dict],
    thorough: bool = False,
    expect: ValidationExpectations = None,
) -> list[ValidationMessage]:
    messages = []

    # TODO: follow ODC's match rules?

    matched_product = None

    if products:
        matched_product, messages = _match_product(doc, products)
    else:
        messages.append(
            ValidationMessage(
                Level.error if thorough else Level.info,
                "no_product",
                "No product provided: validating dataset information alone",
            )
        )

    metadata_type = None
    if metadata_types and matched_product:
        metadata_type = matched_product["metadata_type"]
        if metadata_type not in metadata_types:
            messages.append(
                ValidationMessage(
                    Level.error if thorough else Level.info,
                    "no_metadata_type",
                    f"Metadata type not provided {metadata_type}: not validating fields",
                )
            )

    messages.extend(
        validate_dataset(
            doc,
            product_definition=matched_product,
            readable_location=location,
            thorough=thorough,
            metadata_type_definition=metadata_types.get(metadata_type),
            expect=expect,
        )
    )
    return messages


def _get_printable_differences(dict1: dict, dict2: dict):
    """
    Get a series of lines to print that show the reason that dict1 is not a superset of dict2
    """
    dict1 = dict(utils.flatten_dict(dict1))
    dict2 = dict(utils.flatten_dict(dict2))

    for path in dict2.keys():
        v1, v2 = dict1.get(path), dict2.get(path)
        if v1 != v2:
            yield f"{path}: {v1!r} != {v2!r}"


def _get_product_mismatch_reasons(dataset_doc: dict, product_definition: dict):
    """
    Which fields don't match the given dataset doc to a product definition?

    Gives human-readable lines of text.
    """
    yield from _get_printable_differences(dataset_doc, product_definition["metadata"])


def _match_product(
    dataset_doc: dict, product_definitions: dict[str, dict]
) -> tuple[dict | None, list[ValidationMessage]]:
    """Match the given dataset to a product definition"""

    product = None

    # EO3 datasets often put the product name directly inside.
    specified_product_name = get_path(dataset_doc, ("product", "name"), default=None)
    specified_product_name = specified_product_name or get_path(
        dataset_doc, ("properties", "odc:product"), default=None
    )

    if specified_product_name and (specified_product_name in product_definitions):
        product = product_definitions[specified_product_name]

    matching_products = {
        name: definition
        for name, definition in product_definitions.items()
        if changes.contains(dataset_doc, definition["metadata"])
    }

    # We we have nothing, give up!
    if (not matching_products) and (not product):
        # Find the product that most closely matches it, to helpfully show the differences!
        closest_product_name = None
        closest_differences = None
        for name, definition in product_definitions.items():
            diffs = tuple(_get_product_mismatch_reasons(dataset_doc, definition))
            if (closest_differences is None) or len(diffs) < len(closest_differences):
                closest_product_name = name
                closest_differences = diffs

        difference_hint = _differences_as_hint(closest_differences)
        return None, [
            _error(
                "unknown_product",
                "Dataset does not match the given products",
                hint=f"Closest match is {closest_product_name}, with differences:"
                f"\n{difference_hint}",
            )
        ]

    messages = []

    if specified_product_name not in matching_products:
        if product:
            difference_hint = _differences_as_hint(
                _get_product_mismatch_reasons(dataset_doc, product)
            )
            messages.append(
                _info(
                    "strange_product_claim",
                    f"Dataset claims to be product {specified_product_name!r}, but doesn't match its fields",
                    hint=f"{difference_hint}",
                )
            )
        else:
            messages.append(
                _info(
                    "unknown_product_claim",
                    f"Dataset claims to be product {specified_product_name!r}, but it wasn't supplied.",
                )
            )

    if len(matching_products) > 1:
        matching_names = ", ".join(matching_products.keys())
        messages.append(
            _error(
                "product_match_clash",
                "Multiple products match the given dataset",
                hint=(
                    f"Maybe you need more fields in the 'metadata' section?\n"
                    f"Claims to be a {specified_product_name!r}, and matches {matching_names!r}"
                    if specified_product_name
                    else f"Maybe you need more fields in the 'metadata' section?\n"
                    f"Matches {matching_names!r}"
                ),
            )
        )
        # (We wont pick one from the bunch here. Maybe they already matched one above to use in continuing validation.)

    # Just like ODC, match rules will rule all. Even if their metadata has a "product_name" field.
    if len(matching_products) == 1:
        [product] = matching_products.values()

    return product, messages


def _differences_as_hint(product_diffs):
    return indent("\n".join(product_diffs), prefix="\t")


def _validate_stac_properties(dataset: DatasetDoc):
    for name, value in dataset.properties.items():
        if name not in dataset.properties.KNOWN_PROPERTIES:
            yield _warning("unknown_property", f"Unknown stac property {name!r}")

        else:
            normaliser = dataset.properties.KNOWN_PROPERTIES.get(name)
            if normaliser and value is not None:
                try:
                    normalised_value = normaliser(value)
                    # A normaliser can return two values, the latter adding extra extracted fields.
                    if isinstance(normalised_value, tuple):
                        normalised_value = normalised_value[0]

                    # It's okay for datetimes to be strings
                    # .. since ODC's own loader does that.
                    if isinstance(normalised_value, datetime) and isinstance(
                        value, str
                    ):
                        value = ciso8601.parse_datetime(value)

                    # Special case for dates, as "no timezone" and "utc timezone" are treated identical.
                    if isinstance(value, datetime):
                        value = default_utc(value)

                    if not isinstance(value, type(normalised_value)):
                        yield _warning(
                            "property_type",
                            f"Value {value} expected to be "
                            f"{type(normalised_value).__name__!r} (got {type(value).__name__!r})",
                        )
                    elif normalised_value != value:
                        if _is_nan(normalised_value) and _is_nan(value):
                            # Both are NaNs, ignore.
                            pass
                        else:
                            yield _warning(
                                "property_formatting",
                                f"Property {value!r} expected to be {normalised_value!r}",
                            )
                except ValueError as e:
                    yield _error("invalid_property", f"{name!r}: {e.args[0]}")

    if "odc:producer" in dataset.properties:
        producer = dataset.properties["odc:producer"]
        # We use domain name to avoid arguing about naming conventions ('ga' vs 'geoscience-australia' vs ...)
        if "." not in producer:
            yield _warning(
                "producer_domain",
                "Property 'odc:producer' should be the organisation's domain name. Eg. 'ga.gov.au'",
            )

    # This field is a little odd, but is expected by the current version of ODC.
    # (from discussion with Kirill)
    if not dataset.properties.get("odc:file_format"):
        yield _warning(
            "global_file_format",
            "Property 'odc:file_format' is empty",
            hint="Usually 'GeoTIFF'",
        )


def _is_nan(v):
    # Due to JSON serialisation, nan can also be represented as a string 'NaN'
    if isinstance(v, str):
        return v == "NaN"
    return isinstance(v, float) and math.isnan(v)


def _validate_geo(dataset: DatasetDoc, expect_geometry: bool = True):
    has_some_geo = _has_some_geo(dataset)
    if not has_some_geo and expect_geometry:
        yield _info("non_geo", "No geo information in dataset")
        return

    if dataset.geometry is None:
        if expect_geometry:
            yield _info("incomplete_geo", "Dataset has some geo fields but no geometry")
    elif not dataset.geometry.is_valid:
        yield _error(
            "invalid_geometry",
            f"Geometry is not a valid shape: {explain_validity(dataset.geometry)!r}",
        )

    # TODO: maybe we'll allow no grids: backwards compat with old metadata.
    if not dataset.grids:
        yield _error("incomplete_grids", "Dataset has some geo fields but no grids")

    if not dataset.crs:
        yield _error("incomplete_crs", "Dataset has some geo fields but no crs")
    else:
        # We only officially support epsg code (recommended) or wkt.
        if dataset.crs.lower().startswith("epsg:"):
            try:
                CRS.from_string(dataset.crs)
            except CRSError as e:
                yield _error("invalid_crs_epsg", e.args[0])

            if dataset.crs.lower() != dataset.crs:
                yield _warning("mixed_crs_case", "Recommend lowercase 'epsg:' prefix")
        else:
            wkt_crs = None
            try:
                wkt_crs = CRS.from_wkt(dataset.crs)
            except CRSError as e:
                yield _error(
                    "invalid_crs",
                    f"Expect either an epsg code or a WKT string: {e.args[0]}",
                )

            if wkt_crs and wkt_crs.is_epsg_code:
                yield _warning(
                    "non_epsg",
                    f"Prefer an EPSG code to a WKT when possible. (Can change CRS to 'epsg:{wkt_crs.to_epsg()}')",
                )


def _has_some_geo(dataset: DatasetDoc) -> bool:
    return dataset.geometry is not None or dataset.grids or dataset.crs


def display_result_console(
    url: str, is_valid: bool, messages: list[ValidationMessage], quiet=False
):
    """
    Print validation messages to the Console (using colour if available).
    """
    # Otherwise console output, with color if possible.
    if messages or not quiet:
        echo(f"{bool_style(is_valid)} {url}")

    for message in messages:
        hint = ""
        if message.hint:
            # Indent the hint if it's multi-line.
            if "\n" in message.hint:
                hint = "\t\tHint:\n"
                hint += indent(message.hint, "\t\t" + (" " * 5))
            else:
                hint = f"\t\t(Hint: {message.hint})"
        s = {
            Level.info: dict(),
            Level.warning: dict(fg="yellow"),
            Level.error: dict(fg="red"),
        }
        displayable_code = style(f"{message.code}", **s[message.level], bold=True)
        echo(f"\t{message.level.name[0].upper()} {displayable_code} {message.reason}")
        if hint:
            echo(hint)


def display_result_github(url: str, is_valid: bool, messages: list[ValidationMessage]):
    """
    Print validation messages using Github Action's command language for warnings/errors.
    """
    echo(f"{bool_style(is_valid)} {url}")
    for message in messages:
        hint = ""
        if message.hint:
            # Indent the hint if it's multi-line.
            if "\n" in message.hint:
                hint = "\n\nHint:\n"
                hint += indent(message.hint, (" " * 5))
            else:
                hint = f"\n\n(Hint: {message.hint})"

        if message.level == Level.error:
            code = "::error"
        else:
            code = "::warning"

        text = f"{message.reason}{hint}"

        # URL-Encode any newlines
        text = text.replace("\n", "%0A")
        # TODO: Get the real line numbers?
        echo(f"{code} file={url},line=1::{text}")


_OUTPUT_WRITERS = dict(
    plain=display_result_console,
    quiet=partial(display_result_console, quiet=True),
    github=display_result_github,
)


@click.command(
    help=__doc__
    + """
Paths can be products, dataset documents, or directories to scan (for files matching
names '*.odc-metadata.yaml' etc), either local or URLs.

Datasets are validated against matching products that have been scanned already, so specify
products first, and datasets later, to ensure they can be matched.
"""
)
@click.version_option()
@click.argument("paths", nargs=-1)
@click.option(
    "--warnings-as-errors",
    "-W",
    "strict_warnings",
    is_flag=True,
    help="Fail if any warnings are produced",
)
@click.option(
    "-f",
    "--output-format",
    help="Output format",
    type=click.Choice(list(_OUTPUT_WRITERS)),
    # Are we in Github Actions?
    # Send any warnings/errors in its custom format
    default="github" if "GITHUB_ACTIONS" in os.environ else "plain",
    show_default=True,
)
@click.option(
    "--thorough",
    is_flag=True,
    help="Attempt to read the data/measurements, and check their properties match",
)
@click.option(
    "--explorer-url",
    "explorer_url",
    help="Use product definitions from the given Explorer URL to validate datasets. "
    'Eg: "https://explorer.dea.ga.gov.au/"',
)
@click.option(
    "--odc",
    "use_datacube",
    is_flag=True,
    help="Use product definitions from datacube to validate datasets",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Only print problems, one per line",
)
def run(
    paths: list[str],
    strict_warnings,
    quiet,
    thorough: bool,
    explorer_url: str,
    use_datacube: bool,
    output_format: str,
):
    expect = ValidationExpectations()
    validation_counts: Counter[Level] = collections.Counter()
    invalid_paths = 0
    current_location = Path(".").resolve().as_uri() + "/"

    product_definitions = _load_remote_product_definitions(use_datacube, explorer_url)

    if output_format == "plain" and quiet:
        output_format = "quiet"
    write_file_report = _OUTPUT_WRITERS[output_format]

    for url, messages in validate_paths(
        paths,
        thorough=thorough,
        expect=expect,
        product_definitions=product_definitions,
    ):
        if url.startswith(current_location):
            url = url[len(current_location) :]

        levels = collections.Counter(m.level for m in messages)
        is_invalid = levels[Level.error] > 0
        if strict_warnings:
            is_invalid |= levels[Level.warning] > 0

        if quiet:
            # Errors/Warnings only. Remove info-level.
            messages = [m for m in messages if m.level != Level.info]

        if is_invalid:
            invalid_paths += 1

        for message in messages:
            validation_counts[message.level] += 1

        write_file_report(
            url=url,
            is_valid=not is_invalid,
            messages=messages,
        )

    # Print a summary on stderr for humans.
    if not quiet:
        result = (
            style("failure", fg="red", bold=True)
            if invalid_paths > 0
            else style("valid", fg="green", bold=True)
        )
        secho(f"\n{result}: ", nl=False, err=True)
        if validation_counts:
            echo(
                ", ".join(
                    f"{v} {k.name}{'s' if v > 1 else ''}"
                    for k, v in validation_counts.items()
                ),
                err=True,
            )
        else:
            secho(f"{len(paths)} paths", err=True)

    sys.exit(invalid_paths)


def _load_remote_product_definitions(
    from_datacube: bool = False,
    from_explorer_url: str | None = None,
) -> dict[str, dict]:
    product_definitions = {}
    # Load any remote products that were asked for.
    if from_explorer_url:
        for definition in _load_explorer_product_definitions(from_explorer_url):
            product_definitions[definition["name"]] = definition
        secho(f"{len(product_definitions)} Explorer products", err=True)

    if from_datacube:
        # The normal datacube environment variables can be used to choose alternative configs.
        with Datacube(app="eo3-validate") as dc:
            for product in dc.index.products.get_all():
                product_definitions[product.name] = product.definition

        secho(f"{len(product_definitions)} ODC products", err=True)
    return product_definitions


def _load_doc(url):
    return list(load_documents(url))


def _load_explorer_product_definitions(
    explorer_url: str,
    workers: int = 6,
) -> Generator[dict, None, None]:
    """
    Read all product yamls from the given Explorer instance,

    eg: https://explorer.dea.ga.gov.au/products/ls5_fc_albers.odc-product.yaml
    """
    product_urls = [
        urljoin(explorer_url, f"/products/{name.strip()}.odc-product.yaml")
        for name in urlopen(urljoin(explorer_url, "products.txt"))  # nosec
        .read()
        .decode("utf-8")
        .split("\n")
    ]
    count = 0
    with multiprocessing.Pool(workers) as pool:
        for product_definitions in pool.imap_unordered(_load_doc, product_urls):
            count += 1
            echo(f"\r{count} Explorer products", nl=False)
            yield from product_definitions
        pool.close()
        pool.join()
    echo()
