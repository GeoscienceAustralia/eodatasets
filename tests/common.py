import operator
from collections.abc import Iterable
from pathlib import Path
from textwrap import indent

import pytest
import rapidjson
from click.testing import CliRunner, Result
from deepdiff import DeepDiff
from deepdiff.model import DiffLevel
from ruamel import yaml
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from eodatasets3 import DatasetDoc, serialise


def check_prepare_outputs(
    invoke_script,
    run_args,
    expected_doc: dict,
    expected_metadata_path: Path,
    ignore_fields=(),
):
    """Call a prepare script and check for an expected output document."""
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
    res = run_prepare_cli(invoke_script, *run_args)

    try:
        assert_expected_eo3_path(expected_doc, expected_metadata_path, ignore_fields)
    except AssertionError:
        print(f"Output:\n{indent(res.output, '    ')}")
        raise


def assert_expected_eo3_path(
    expected_doc: dict,
    expected_path: Path,
    ignore_fields=(),
):
    """
    Check an output path of an EO3 dataset matches an expected document.

    This is slightly smarter about doing geometry equality etc within the document.
    """
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
    assert expected_path.exists(), (
        f"Expected output EO3 path doesn't exist: {expected_path}"
    )
    assert_same_as_file(
        expected_doc,
        expected_path,
        # We check the geometry below
        ignore_fields=("geometry", *ignore_fields),
    )

    if "geometry" not in ignore_fields:
        # Compare geometry after parsing, rather than comparing the raw dict values.
        produced_dataset = serialise.from_path(expected_path)
        expected_dataset = serialise.from_doc(expected_doc, skip_validation=True)
        if expected_dataset.geometry is None:
            assert produced_dataset.geometry is None, (
                f"Expected a null geometry, "
                f"but output included one: {produced_dataset.geometry.__geo_interface__!r}"
            )
        else:
            assert_shapes_mostly_equal(
                produced_dataset.geometry,
                expected_dataset.geometry,
                # Typically meters -- this is easily good enough accuracy.
                0.0001,
            )


def assert_expected_eo3(
    expected_doc: DatasetDoc,
    given_doc: DatasetDoc,
    *,
    ignore_fields=(),
):
    """
    Do the two DatasetDocs match?

    (Unlike equality, gives reasonable error message of differences, and
    compares geometry more intelligently.)
    """
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
    if expected_doc.geometry is None:
        assert given_doc.geometry is None, "Expected no geometry"
    else:
        assert_shapes_mostly_equal(
            given_doc.geometry, expected_doc.geometry, 0.00000001
        )
    e = serialise.to_doc(expected_doc)
    g = serialise.to_doc(given_doc)
    for f in ("geometry", *ignore_fields):
        e.pop(f)
        g.pop(f)
    assert_same(g, e)


def assert_shapes_mostly_equal(
    shape1: BaseGeometry | dict,
    shape2: BaseGeometry | dict,
    threshold: float,
):
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)

    if isinstance(shape1, dict):
        shape1 = shape(shape1)
    if isinstance(shape2, dict):
        shape2 = shape(shape2)

    # Check area first, as it's a nicer error message when they're wildly different.
    assert shape1.area == pytest.approx(shape2.area, abs=threshold), (
        f"Shapes have different areas: {shape1.area} != {shape2.area}"
    )

    s1 = shape1.simplify(tolerance=threshold)
    s2 = shape2.simplify(tolerance=threshold)
    assert (s1 - s2).area < threshold, f"{s1} is not mostly equal to {s2}"


def assert_same(expected_doc: dict, generated_doc: dict):
    """
    Assert two documents are the same, ignoring trivial float differences
    """
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)
    doc_diffs = DeepDiff(expected_doc, generated_doc, significant_digits=6)
    assert doc_diffs == {}, "\n".join(format_doc_diffs(expected_doc, generated_doc))


def assert_same_as_file(expected_doc: dict, generated_file: Path, ignore_fields=()):
    """Assert a file contains the given document content (after normalisation etc)"""
    __tracebackhide__ = operator.methodcaller("errisinstance", AssertionError)

    assert generated_file.exists(), f"Expected file to exist {generated_file.name}"

    generated_doc = load_yaml(generated_file)

    expected_doc = dict(expected_doc)
    for field in ignore_fields:
        if field in generated_doc:
            del generated_doc[field]
        if field in expected_doc:
            del expected_doc[field]

    expected_doc = dump_roundtrip(expected_doc)
    generated_doc = dump_roundtrip(generated_doc)
    assert_same(generated_doc, expected_doc)


def load_yaml(generated_file):
    with generated_file.open("r") as f:
        generated_doc = yaml.YAML(typ="safe").load(f)
    return generated_doc


def run_prepare_cli(invoke_script, *args, expect_success=True) -> Result:
    """Run the prepare script as a command-line command"""
    __tracebackhide__ = True

    res: Result = CliRunner().invoke(
        invoke_script, [str(a) for a in args], catch_exceptions=False
    )

    if expect_success:
        assert res.exit_code == 0, f"Failed with output: {res.output}"

    return res


def format_doc_diffs(left: dict, right: dict) -> Iterable[str]:
    """
    Get a human-readable list of differences in the given documents.

    Returns a list of lines to print.
    """
    doc_diffs = DeepDiff(left, right, significant_digits=6)
    out = []
    if doc_diffs:
        out.append("Documents differ:")
    else:
        out.append("Doc differs in minor float precision:")
        doc_diffs = DeepDiff(left, right)

    def clean_offset(offset: str):
        if offset.startswith("root"):
            return offset[len("root") :]
        return offset

    if "values_changed" in doc_diffs:
        for offset, change in doc_diffs["values_changed"].items():
            out.extend(
                (
                    f"   {clean_offset(offset)}: ",
                    f"          {change['old_value']!r}",
                    f"       != {change['new_value']!r}",
                )
            )
    if "dictionary_item_added" in doc_diffs:
        out.append("Added fields:")
        for offset in doc_diffs.tree["dictionary_item_added"]:
            offset: DiffLevel
            out.append(f"    {clean_offset(offset.path())} = {offset.t2!r}")
    if "dictionary_item_removed" in doc_diffs:
        out.append("Removed fields:")
        for offset in doc_diffs.tree["dictionary_item_removed"]:
            offset: DiffLevel
            out.append(f"    {clean_offset(offset.path())} = {offset.t1!r}")
    # Anything we missed from the (sometimes changing) diff api?
    if len(out) == 1:
        out.append(repr(doc_diffs))

    # If pytest verbose:
    out.extend(("Full output document: ", repr(left)))
    return out


def dump_roundtrip(generated_doc):
    """Do a dump/load to normalise all doc-neutral dict/date/tuple/list types.

    The in-memory choice of dict/etc subclasses shouldn't matter, as long as the doc
    is identical once produced.
    """
    return rapidjson.loads(
        rapidjson.dumps(generated_doc, datetime_mode=True, uuid_mode=True)
    )
