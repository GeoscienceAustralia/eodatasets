"""
Module
"""

import binascii
import hashlib
import tempfile
from pathlib import Path

import numpy
import rasterio
from rasterio import DatasetReader

allow_anything = object()


def assert_image(
    image: Path,
    overviews=allow_anything,
    nodata=allow_anything,
    unique_pixel_counts: dict = allow_anything,
    bands=1,
    shape: tuple[int, int] = None,
):
    __tracebackhide__ = True
    with rasterio.open(image) as d:
        d: DatasetReader
        assert d.count == bands, f"Expected {bands} band{'s' if bands > 1 else ''}"

        if overviews is not allow_anything:
            assert d.overviews(1) == overviews, (
                f"Unexpected overview: {d.overviews(1)!r} != {overviews!r}"
            )
        if nodata is not allow_anything:
            assert d.nodata == nodata, f"Unexpected nodata: {d.nodata!r} != {nodata!r}"

        if unique_pixel_counts is not allow_anything:
            array = d.read(1)
            value_counts = dict(zip(*numpy.unique(array, return_counts=True)))
            assert value_counts == unique_pixel_counts, (
                f"Unexpected pixel counts: {value_counts!r} != {unique_pixel_counts!r}"
            )

        if shape:
            assert shape == d.shape, f"Unexpected shape: {shape!r} != {d.shape!r}"


def load_checksum_filenames(output_metadata_path):
    return [
        line.split("\t")[-1][:-1] for line in output_metadata_path.open("r").readlines()
    ]


def on_same_filesystem(path1, path2):
    return path1.stat().st_dev == path2.stat().st_dev


def hardlink_arg(path1, path2):
    return "--hard-link" if on_same_filesystem(path1, path2) else "--no-hard-link"


def directory_size(directory):
    """
    Total size of files in the given directory.
    :type file_paths: Path
    :rtype: int
    """
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


class FakeAncilFile:
    def __init__(self, base_folder, type_, filename, folder_offset=()):
        """
        :type base_folder: pathlib.Path
        :type type_: str
        :type filename: str
        :type folder_offset: tuple[str]
        :return:
        """
        self.base_folder = base_folder
        self.type_ = type_
        self.filename = filename
        self.folder_offset = folder_offset

    def create(self):
        """Create our dummy ancillary file"""
        self.containing_folder.mkdir(parents=True)
        with self.file_path.open("wb") as f:
            # Write the file path into it so that it has a unique checksum.
            f.write(str(self.file_path).encode("utf8"))

    @property
    def checksum(self):
        m = hashlib.sha1()
        m.update(str(self.file_path).encode("utf8"))
        return binascii.hexlify(m.digest()).decode("ascii")

    @property
    def containing_folder(self):
        return self.base_folder.joinpath(self.type_, *self.folder_offset)

    @property
    def file_path(self):
        return self.containing_folder.joinpath(self.filename)


def prepare_work_order(ancil_files, work_order_template_path):
    """

    :type ancil_files: tuple[FakeAncilFile]
    :type work_order_template_path: pathlib.Path
    :rtype: pathlib.Path
    """
    # Create the dummy Ancil files.
    for ancil in ancil_files:
        ancil.create()

    work_dir = Path(tempfile.mkdtemp())
    # Write a work order with ancillary locations replaced.
    output_work_order = work_dir.joinpath("work_order.xml")
    with work_order_template_path.open("rb") as wo:
        wo_text = (
            wo.read()
            .decode("utf-8")
            .format(**{a.type_ + "_path": a.file_path for a in ancil_files})
        )
        with output_work_order.open("w") as out_wo:
            out_wo.write(wo_text)

    return output_work_order
