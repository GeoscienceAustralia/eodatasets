import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from typing import IO
from uuid import UUID

import attr
import cattr
import ciso8601
import click
import jsonschema
import numpy
import shapely
import shapely.affinity
import shapely.ops
from affine import Affine
from datacube.model import SCHEMA_PATH as DATACUBE_SCHEMAS_PATH
from datacube.utils import read_documents
from ruamel.yaml import YAML, Representer
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from eodatasets3.model import ODC_DATASET_SCHEMA_URL, DatasetDoc, Eo3Dict
from eodatasets3.properties import FileFormat

converter = cattr.Converter()


def _format_representer(dumper, data: FileFormat):
    return dumper.represent_scalar("tag:yaml.org,2002:str", f"{data.name}")


def _uuid_representer(dumper, data):
    """
    :type dumper: yaml.representer.BaseRepresenter
    :type data: uuid.UUID
    :rtype: yaml.nodes.Node
    """
    return dumper.represent_scalar("tag:yaml.org,2002:str", f"{data}")


def _represent_datetime(self, data: datetime):
    """
    The default Ruamel representer strips 'Z' suffixes for UTC.

    But we like to be explicit.
    """
    # If there's a non-utc timezone, use it.
    if data.tzinfo is not None and (data.utcoffset().total_seconds() > 0):
        value = data.isoformat(" ")
    else:
        # Otherwise it's UTC (including when tz==null).
        value = data.replace(tzinfo=None).isoformat(" ") + "Z"
    return self.represent_scalar("tag:yaml.org,2002:timestamp", value)


def _represent_numpy_datetime(self, data: numpy.datetime64):
    return _represent_datetime(self, data.astype("M8[ms]").tolist())


def _represent_paths(self, data: PurePath):
    return Representer.represent_str(self, data.as_posix())


def _represent_float(self, data: float):
    float_text = numpy.format_float_scientific(data)
    return self.represent_scalar("tag:yaml.org,2002:float", float_text)


def _init_yaml() -> YAML:
    yaml = YAML()

    yaml.representer.add_representer(FileFormat, _format_representer)
    yaml.representer.add_multi_representer(UUID, _uuid_representer)
    yaml.representer.add_representer(datetime, _represent_datetime)
    yaml.representer.add_multi_representer(PurePath, _represent_paths)

    # WAGL spits out many numpy primitives in docs.
    yaml.representer.add_representer(numpy.int8, Representer.represent_int)
    yaml.representer.add_representer(numpy.uint8, Representer.represent_int)
    yaml.representer.add_representer(numpy.int16, Representer.represent_int)
    yaml.representer.add_representer(numpy.uint16, Representer.represent_int)
    yaml.representer.add_representer(numpy.int32, Representer.represent_int)
    yaml.representer.add_representer(numpy.uint32, Representer.represent_int)
    yaml.representer.add_representer(numpy.int64, Representer.represent_int)
    yaml.representer.add_representer(numpy.uint64, Representer.represent_int)
    # Representer.represent_float is currently incompatible with numpy2 floats
    yaml.representer.add_representer(numpy.float32, _represent_float)
    yaml.representer.add_representer(numpy.float64, _represent_float)

    yaml.representer.add_representer(numpy.ndarray, Representer.represent_list)
    yaml.representer.add_representer(numpy.datetime64, _represent_numpy_datetime)

    # Match yamllint default expectations. (Explicit start/end are recommended to tell if a file is cut off)
    yaml.width = 80
    yaml.explicit_start = True
    yaml.explicit_end = True

    return yaml


def dump_yaml(output_yaml: Path, *docs: Mapping) -> None:
    if not output_yaml.name.lower().endswith(".yaml"):
        raise ValueError(
            f"YAML filename doesn't end in *.yaml (?). Received {output_yaml!r}"
        )

    yaml = _init_yaml()
    with output_yaml.open("w") as stream:
        yaml.dump_all(docs, stream)


def dumps_yaml(stream, *docs: Mapping) -> None:
    """Dump yaml through a stream, using the default serialisation settings."""
    yml = _init_yaml()
    yml.representer.add_representer(float, _represent_float)
    return yml.dump_all(docs, stream=stream)


def load_yaml(p: Path) -> dict:
    with p.open() as f:
        return _yaml().load(f)


def _yaml():
    return YAML(typ="safe")


def loads_yaml(stream: str | IO) -> Iterable[dict]:
    """Dump yaml through a stream, using the default deserialisation settings."""
    return _yaml().load_all(stream)


def from_path(path: Path, skip_validation=False) -> DatasetDoc:
    """
    Parse an EO3 document from a filesystem path

    :param path: Filesystem path
    :param skip_validation: Optionally disable validation (it's faster, but I hope your
            doc is structured correctly)
    """
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"Unexpected file type {path.suffix}. Expected yaml")

    return from_doc(load_yaml(path), skip_validation=skip_validation)


class InvalidDataset(Exception):
    def __init__(self, path: Path, error_code: str, reason: str) -> None:
        self.path = path
        self.error_code = error_code
        self.reason = reason


def _is_json_array(checker, instance) -> bool:
    """
    By default, jsonschema only allows a json array to be a Python list.
    Let's allow it to be a tuple too.
    """
    return isinstance(instance, list | tuple)


def _load_schema_validator(p: Path) -> jsonschema.Draft7Validator:
    """
    Create a schema instance for the file.

    (Assumes they are trustworthy. Only local schemas!)
    """
    import referencing

    if not p.is_file():
        raise ValueError(f"Can only load local schemas. Could not find file {p!s}")
    if p.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"Unexpected file type {p.suffix}. Expected yaml")
    with p.open() as f:
        schema = _yaml().load(f)
    jsonschema.Draft7Validator.check_schema(schema)

    # Allow schemas to reference other schemas relatively
    def doc_reference(path):
        path = p.parent.joinpath(path)
        if not path.exists():
            raise ValueError(f"Reference not found: {path}")
        referenced_schema = next(iter(read_documents(path)))[1]
        return referencing.Resource(referenced_schema, referencing.jsonschema.DRAFT7)

    if p.parent:
        registry = referencing.Registry(retrieve=doc_reference)  # type: ignore[call-arg, var-annotated]
    else:
        registry = referencing.Registry()

    validator = jsonschema.validators.extend(
        jsonschema.Draft7Validator,
        type_checker=jsonschema.Draft7Validator.TYPE_CHECKER.redefine(
            "array", _is_json_array
        ),
    )
    return validator(schema, registry=registry)


DATASET_SCHEMA = _load_schema_validator(Path(__file__).parent / "dataset.schema.yaml")
PRODUCT_SCHEMA = _load_schema_validator(
    DATACUBE_SCHEMAS_PATH / "dataset-type-schema.yaml"
)
METADATA_TYPE_SCHEMA = _load_schema_validator(
    DATACUBE_SCHEMAS_PATH / "metadata-type-schema.yaml"
)


def from_doc(doc: dict, skip_validation=False) -> DatasetDoc:
    """
    Parse a dictionary into an EO3 dataset.

    By default it will validate it against the schema, which will result in far more
    useful error messages if fields are missing.

    :param doc: A dictionary, such as is returned from yaml.load or json.load
    :param skip_validation: Optionally disable validation (it's faster, but I hope your
            doc is structured correctly)
    """
    doc = doc.copy()
    if not skip_validation:
        # don't error if properties 'extent' or 'grid_spatial' are present
        if doc.get("extent"):
            del doc["extent"]
        if doc.get("grid_spatial"):
            del doc["grid_spatial"]
        DATASET_SCHEMA.validate(doc)

    # TODO: stable cattrs (<1.0) balks at the $schema variable.
    del doc["$schema"]
    location = doc.pop("location", None)
    if location:
        doc["locations"] = [location]

    return converter.structure(doc, DatasetDoc)


def _structure_as_uuid(d, t):
    return uuid.UUID(str(d))


def _structure_as_stac_props(d, t, normalise_properties=False):
    """
    :param normalise_properties:
        We don't normalise properties by default as we usually want it to reflect the original file.

    """
    return Eo3Dict(
        # The passed-in dictionary is stored internally, so we want to make a copy of it
        # so that our serialised output is fully separate from the input.
        dict(d),
        normalise_input=normalise_properties,
    )


def _structure_as_affine(d: tuple, t):
    if len(d) not in [6, 9]:
        raise ValueError(f"Expected 6 or 9 coefficients in transform. Got {d!r}")

    if len(d) == 9:
        if tuple(d[-3:]) != (0.0, 0.0, 1.0):
            raise ValueError(
                f"Nine-element affine should always end in [0, 0, 1]. Got {d!r}"
            )
        d = [*d[:-3]]

    return Affine(*d)


def _unstructure_as_stac_props(v: Eo3Dict):
    return v._props


def _structure_as_shape(d, t):
    return shape(d)


converter.register_structure_hook(uuid.UUID, _structure_as_uuid)
converter.register_structure_hook(BaseGeometry, _structure_as_shape)
converter.register_structure_hook(
    Eo3Dict,
    partial(_structure_as_stac_props, normalise_properties=False),
)
converter.register_structure_hook(Affine, _structure_as_affine)
converter.register_unstructure_hook(Eo3Dict, _unstructure_as_stac_props)


def to_doc(d: DatasetDoc) -> dict:
    """
    Serialise a DatasetDoc to a dict

    If you plan to write this out as a yaml file on disk, you're
    better off with one of our formatted writers: :func:`.to_stream`, :func:`.to_path`.
    """
    doc = attr.asdict(
        d,
        recurse=True,
        dict_factory=dict,
        # Exclude fields that are the default.
        filter=lambda attr, value: "doc_exclude" not in attr.metadata
        and value != attr.default
        # Exclude any fields set to None. The distinction should never matter in our docs.
        and value is not None,
        retain_collection_types=False,
    )
    doc["$schema"] = ODC_DATASET_SCHEMA_URL
    if d.geometry is not None:
        doc["geometry"] = shapely.geometry.mapping(d.geometry)
    doc["id"] = str(d.id)
    doc["properties"] = dict(d.properties)

    if len(doc.get("locations", [])) == 1:
        doc["location"] = doc.pop("locations")[0]

    return doc


def to_formatted_doc(d: DatasetDoc) -> CommentedMap:
    """Serialise a DatasetDoc to a yaml-serialisation-ready dict"""
    doc = prepare_formatting(to_doc(d))
    # Add user-readable names for measurements as a comment if present.
    if d.measurements:
        for band_name, band_doc in d.measurements.items():
            if band_doc.alias and band_name.lower() != band_doc.alias.lower():
                doc["measurements"].yaml_add_eol_comment(band_doc.alias, band_name)

    return doc


def to_path(path: Path, *ds: DatasetDoc):
    """
    Output dataset(s) as a formatted YAML to a local path

    (multiple datasets will result in a multi-document yaml file)
    """
    dump_yaml(path, *(to_formatted_doc(d) for d in ds))


def to_stream(stream, *ds: DatasetDoc):
    """
    Output dataset(s) as a formatted YAML to an output stream

    (multiple datasets will result in a multi-document yaml file)
    """
    dumps_yaml(stream, *(to_formatted_doc(d) for d in ds))


def _stac_key_order(key: str):
    """All keys in alphabetical order, but unprefixed keys first."""
    if ":" in key:
        # Tilde comes after all alphanumerics.
        return f"~{key}"
    else:
        return key


def _eo3_key_order(keyval: str):
    """
    Order keys in an an EO3 document.

    Suitable for sorted() func usage.
    """
    key, val = keyval
    try:
        i = _EO3_PROPERTY_ORDER.index(key)
        if i == -1:
            return 999
        return i
    except ValueError:
        return 999


# A logical, readable order for properties to be in a dataset document.
_EO3_PROPERTY_ORDER = [
    "$schema",
    # Products / Types
    "name",
    "license",
    "metadata_type",
    "description",
    "metadata",
    # EO3
    "id",
    "label",
    "product",
    "location",
    "locations",
    "crs",
    "geometry",
    "grids",
    "properties",
    "measurements",
    "accessories",
    "lineage",
]


def prepare_formatting(d: Mapping) -> CommentedMap:
    """
    Format an eo3 dataset dict for human-readable yaml serialisation.

    This will order fields, add whitespace, comments, etc.

    Output is intended for ruamel.yaml.
    """
    # Sort properties for readability.
    doc = CommentedMap(sorted(d.items(), key=_eo3_key_order))
    doc["properties"] = CommentedMap(
        sorted(doc["properties"].items(), key=_stac_key_order)
    )

    # Whitespace
    doc.yaml_set_comment_before_after_key("$schema", before="Dataset")
    if "geometry" in doc:
        # Set some numeric fields to be compact yaml format.
        _use_compact_format(doc["geometry"], "coordinates")
    if "grids" in doc:
        for grid in doc["grids"].values():
            _use_compact_format(grid, "shape", "transform")

    _add_space_before(
        doc,
        "label" if "label" in doc else "id",
        "crs",
        "properties",
        "measurements",
        "accessories",
        "lineage",
        "location",
        "locations",
    )

    p: CommentedMap = doc["properties"]
    p.yaml_add_eol_comment("# Ground sample distance (m)", "eo:gsd")

    return doc


def _use_compact_format(d: dict, *keys):
    """Change the given sequence to compact YAML form"""
    for key in keys:
        if key in d:
            d[key] = CommentedSeq(d[key])
            d[key].fa.set_flow_style()


def _add_space_before(d: CommentedMap, *keys):
    """Add an empty line to the document before a section (key)"""
    for key in keys:
        d.yaml_set_comment_before_after_key(key, before="\n")


class ClickDatetime(click.ParamType):
    """
    Take a datetime parameter, supporting any ISO8601 date/time/timezone combination.
    """

    name = "date"

    def convert(self, value, param, ctx):
        if value is None:
            return value

        if isinstance(value, datetime):
            return value

        try:
            return ciso8601.parse_datetime(value)
        except ValueError:
            self.fail(
                (
                    f"Invalid date string {value!r}. Expected any ISO date/time format "
                    '(eg. "2017-04-03" or "2014-05-14 12:34")'
                ),
                param,
                ctx,
            )
