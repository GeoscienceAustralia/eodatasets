import math
import os
import string
import sys
import tempfile
from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from enum import Enum, auto
from pathlib import Path, PurePath

import attr
import numpy
import rasterio
import rasterio.features
import shapely
import shapely.affinity
import shapely.ops
from odc.geo import CRS
from odc.geo.geobox import GeoBox
from rasterio import DatasetReader
from rasterio.enums import Resampling
from rasterio.io import DatasetWriter, MemoryFile
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import binary_fill_holes
from shapely.geometry import box
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE, BaseGeometry

from eodatasets3.model import DatasetDoc, GridDoc, MeasurementDoc

DEFAULT_OVERVIEWS = (8, 16, 32)


class ValidDataMethod(Enum):
    """
    How to calculate the valid data geometry for an image?
    """

    #: Vectorize the full valid pixel mask as-is.
    #:
    #: In some circumstances this can be very slow.
    #: `filled` may be safer.
    #:
    thorough = auto()

    #: Fill holes in the valid pixel mask before vectorizing.
    #:
    #: (Potentially much faster than ``thorough`` if there's many small
    #: nodata holes, as they will create many tiny polygons.
    #: *slightly* slower if no holes exist.)
    filled = auto()

    #: Take convex-hull of valid pixel mask before vectorizing.
    #:
    #: This is much slower than ``filled``, but will work in cases where
    #: you have a lot of internal geometry that aren't holes.
    #: Such as SLC-Off Landsat 7 data.
    #:
    #: Requires 'scikit-image' dependency.
    convex_hull = auto()

    #: Use the image file bounds, ignoring actual pixel values.
    bounds = auto()


def gbox_from_path(path: str) -> GeoBox:
    """Create from the spec of a (rio-readable) filesystem path or url"""
    with rasterio.open(path) as rio:
        return GeoBox.from_rio(rio)


def gbox_from_dataset_doc(ds: DatasetDoc, grid="default") -> GeoBox:
    """
    Create from an existing parsed metadata document

    :param grid: Grid name to read, if not the default.
    """
    g = ds.grids[grid]

    return GeoBox(g.shape, g.transform, crs=ds.crs)


def generate_tiles(
    samples: int, lines: int, xtile: int = None, ytile: int = None
) -> Generator[tuple[tuple[int, int], tuple[int, int]], None, None]:
    """
    Generates a list of tile indices for a 2D array.

    :param samples:
        An integer expressing the total number of samples in an array.

    :param lines:
        An integer expressing the total number of lines in an array.

    :param xtile:
        (Optional) The desired size of the tile in the x-direction.
        Default is all samples

    :param ytile:
        (Optional) The desired size of the tile in the y-direction.
        Default is min(100, lines) lines.

    :return:
        Each tuple in the generator contains
        ((ystart,yend),(xstart,xend)).

    >>> import pprint
    >>> tiles = generate_tiles(1624, 1567, xtile=1000, ytile=400)
    >>> pprint.pprint(list(tiles))
    [((0, 400), (0, 1000)),
     ((0, 400), (1000, 1624)),
     ((400, 800), (0, 1000)),
     ((400, 800), (1000, 1624)),
     ((800, 1200), (0, 1000)),
     ((800, 1200), (1000, 1624)),
     ((1200, 1567), (0, 1000)),
     ((1200, 1567), (1000, 1624))]
    """

    def create_tiles(samples, lines, xstart, ystart):
        """
        Creates a generator object for the tiles.
        """
        for ystep in ystart:
            if ystep + ytile < lines:
                yend = ystep + ytile
            else:
                yend = lines
            for xstep in xstart:
                if xstep + xtile < samples:
                    xend = xstep + xtile
                else:
                    xend = samples
                yield ((ystep, yend), (xstep, xend))

    # check for default or out of bounds
    if xtile is None or xtile < 0:
        xtile = samples
    if ytile is None or ytile < 0:
        ytile = min(100, lines)

    xstart = numpy.arange(0, samples, xtile)
    ystart = numpy.arange(0, lines, ytile)

    tiles = create_tiles(samples, lines, xstart, ystart)

    return tiles


def _common_suffix(names: Iterable[str]) -> str:
    return os.path.commonprefix([s[::-1] for s in names])[::-1]


def _find_a_common_name(
    group_of_names: Sequence[str], all_possible_names: set[str] = None
) -> str | None:
    """
    If we have a list of band names, can we find a nice name for the group of them?

    (used when naming the grid for a set of bands)

    >>> _find_a_common_name(['nbar_blue', 'nbar_red'])
    'nbar'
    >>> _find_a_common_name(['nbar_band08', 'nbart_band08'])
    'band08'
    >>> _find_a_common_name(['nbar:band08', 'nbart:band08'])
    'band08'
    >>> _find_a_common_name(['panchromatic'])
    'panchromatic'
    >>> _find_a_common_name(['nbar_panchromatic'])
    'nbar_panchromatic'
    >>> # It's ok to find nothing.
    >>> _find_a_common_name(['nbar_blue', 'nbar_red', 'qa'])
    >>> _find_a_common_name(['a', 'b'])
    >>> # If a name is taken by non-group memebers, it shouldn't be chosen
    >>> # (There's an 'nbar' prefix outside of the group, so shouldn't be found)
    >>> all_names = {'nbar_blue', 'nbar_red', 'nbar_green', 'nbart_blue'}
    >>> _find_a_common_name(['nbar_blue', 'nbar_red'], all_possible_names=all_names)
    >>> _find_a_common_name(['nbar_blue', 'nbar_red', 'nbar_green'], all_possible_names=all_names)
    'nbar'
    """
    options = []

    non_group_names = (all_possible_names or set()).difference(group_of_names)

    # If all measurements have a common prefix (like 'nbar_') it makes a nice grid name.
    prefix = os.path.commonprefix(group_of_names)
    if not any(name.startswith(prefix) for name in non_group_names):
        options.append(prefix)

    suffix = _common_suffix(group_of_names)
    if not any(name.endswith(suffix) for name in non_group_names):
        options.append(suffix)

    if not options:
        return None

    options = [s.strip("_:") for s in options]
    # Pick the longest candidate.
    options.sort(key=len, reverse=True)
    return options[0] or None


@attr.s(auto_attribs=True, slots=True)
class _MeasurementLocation:
    path: Path | str
    layer: str = None


_Measurements = dict[str, _MeasurementLocation]


class MeasurementBundler:
    """
    Incrementally record the information for a set of measurements/images to group into grids,
    calculate geometry etc, suitable for metadata.
    """

    def __init__(self):
        # The measurements grouped by their grid.
        # (value is band_name->Path)
        self._measurements_per_grid: dict[GeoBox, _Measurements] = defaultdict(dict)
        # Valid data mask per grid, in pixel coordinates.
        self.mask_by_grid: dict[GeoBox, numpy.ndarray] = {}

    def record_image(
        self,
        name: str,
        grid: GeoBox,
        path: PurePath | str,
        img: numpy.ndarray,
        layer: str | None = None,
        nodata: float | int | None = None,
        expand_valid_data=True,
    ):
        for measurements in self._measurements_per_grid.values():
            if name in measurements:
                raise ValueError(
                    f"Duplicate addition of band called {name!r}. "
                    f"Original at {measurements[name]} and now {path}"
                )

        self._measurements_per_grid[grid][name] = _MeasurementLocation(path, layer)
        if expand_valid_data:
            self._expand_valid_data_mask(grid, img, nodata)

    def _expand_valid_data_mask(
        self, grid: GeoBox, img: numpy.ndarray, nodata: float | int
    ):
        if nodata is None:
            nodata = float("nan") if numpy.issubdtype(img.dtype, numpy.floating) else 0

        if math.isnan(nodata):
            valid_values = numpy.isfinite(img)
        else:
            valid_values = img != nodata

        mask = self.mask_by_grid.get(grid)
        if mask is None:
            mask = valid_values
        else:
            mask |= valid_values
        self.mask_by_grid[grid] = mask

    def _as_named_grids(self) -> dict[str, tuple[GeoBox, _Measurements]]:
        """Get our grids with sensible (hopefully!), names."""

        # Order grids from most to fewest measurements.
        # PyCharm's typing seems to get confused by the sorted() call.
        # noinspection PyTypeChecker
        grids_by_frequency: list[tuple[GeoBox, _Measurements]] = sorted(
            self._measurements_per_grid.items(), key=lambda k: len(k[1]), reverse=True
        )

        # The largest group is the default.
        default_grid = grids_by_frequency.pop(0)

        named_grids = {"default": default_grid}

        # No other grids? Nothing to do!
        if not grids_by_frequency:
            return named_grids

        # First try to name them via common prefixes, suffixes etc.
        all_measurement_names = set(self.iter_names())
        for grid, measurements in grids_by_frequency:
            if len(measurements) == 1:
                grid_name = "_".join(measurements.keys())
            else:
                grid_name = _find_a_common_name(
                    list(measurements.keys()), all_possible_names=all_measurement_names
                )
                if not grid_name:
                    # Nothing useful found!
                    break

            if grid_name in named_grids:
                # Clash of names! This strategy wont work.
                break

            named_grids[grid_name] = (grid, measurements)
        else:
            # We finished without a clash.
            return named_grids

        # Otherwise, try resolution names:
        named_grids = {"default": default_grid}
        for grid, measurements in grids_by_frequency:
            res_y, res_x = grid.resolution.map(abs).yx
            if res_x > 1:
                res_x = int(res_x)
            grid_name = f"{res_x}"
            if grid_name in named_grids:
                # Clash of names! This strategy wont work.
                break

            named_grids[grid_name] = (grid, measurements)
        else:
            # We finished without a clash.
            return named_grids

        # No strategies worked!
        # Enumerated, alphabetical letter names. Grid 'a', Grid 'b', etc...
        grid_names = list(string.ascii_letters)
        if len(grids_by_frequency) > len(grid_names):
            raise NotImplementedError(
                f"More than {len(grid_names)} grids that cannot be named!"
            )
        return {
            "default": default_grid,
            **{
                grid_names[i]: (grid, measurements)
                for i, (grid, measurements) in enumerate(grids_by_frequency)
            },
        }

    def as_geo_docs(self) -> tuple[CRS, dict[str, GridDoc], dict[str, MeasurementDoc]]:
        """Calculate combined geo information for metadata docs"""

        if not self._measurements_per_grid:
            return None, None, None

        grid_docs: dict[str, GridDoc] = {}
        measurement_docs: dict[str, MeasurementDoc] = {}
        crs = None

        for grid_name, (grid, measurements) in self._as_named_grids().items():
            # Validate assumption: All grids should have same CRS
            if crs is None:
                crs = grid.crs

            # TODO: CRS equality is tricky. This may not work.
            #       We're assuming a group of measurements specify their CRS
            #       the same way if they are the same.
            elif (grid.crs is not None) and grid.crs != crs:
                raise ValueError(
                    f"Measurements have different CRSes in the same dataset:\n"
                    f"\t{crs.to_string()!r}\n"
                    f"\t{grid.crs.to_string()!r}\n"
                )

            grid_docs[grid_name] = GridDoc(grid.shape.wh, grid.transform)

            for measurement_name, measurement_path in measurements.items():
                # No measurement groups in the doc: we replace with underscores.
                measurement_name = measurement_name.replace(":", "_")

                measurement_docs[measurement_name] = MeasurementDoc(
                    path=measurement_path.path,
                    layer=measurement_path.layer,
                    grid=grid_name if grid_name != "default" else None,
                )
        return crs, grid_docs, measurement_docs

    def consume_and_get_valid_data(
        self, valid_data_method: ValidDataMethod = ValidDataMethod.thorough
    ) -> BaseGeometry:
        """
        Consume the stored grids and produce the valid data for them.

        (they are consumed in order to to minimise peak memory usage)

        :param valid_data_method: How to calculate the valid-data polygon?
        """

        geoms = []

        while self.mask_by_grid:
            grid, mask = self.mask_by_grid.popitem()

            if valid_data_method is ValidDataMethod.bounds:
                geom = box(*grid.boundingbox)
            elif valid_data_method is ValidDataMethod.filled:
                mask = mask.astype("uint8")
                binary_fill_holes(mask, output=mask)
                geom = _grid_to_poly(grid, mask)
            elif valid_data_method is ValidDataMethod.convex_hull:
                # Requires optional dependency scikit-image
                from skimage import morphology as morph

                geom = _grid_to_poly(
                    grid, morph.convex_hull_image(mask).astype("uint8")
                )
            elif valid_data_method is ValidDataMethod.thorough:
                geom = _grid_to_poly(grid, mask.astype("uint8"))
            else:
                raise NotImplementedError(
                    f"Unexpected valid data method: {valid_data_method}"
                )
            geoms.append(geom)
        return shapely.ops.unary_union(geoms)

    def iter_names(self) -> Generator[str, None, None]:
        """All known measurement names"""
        for grid, measurements in self._measurements_per_grid.items():
            for band_name, _ in measurements.items():
                yield band_name

    def iter_paths(self) -> Generator[tuple[GeoBox, str, Path], None, None]:
        """All current measurement paths on disk"""
        for grid, measurements in self._measurements_per_grid.items():
            for band_name, meas_path in measurements.items():
                yield grid, band_name, meas_path.path


def _valid_shape(shape: BaseGeometry) -> BaseGeometry:
    if shape.is_valid:
        return shape
    return shape.buffer(0)


def _grid_to_poly(grid: GeoBox, mask: numpy.ndarray) -> BaseGeometry:
    shape = shapely.ops.unary_union(
        [
            _valid_shape(shapely.geometry.shape(shape))
            for shape, val in rasterio.features.shapes(mask)
            if val == 1
        ]
    )
    shape_y, shape_x = mask.shape
    del mask
    # convex hull
    geom = shape.convex_hull
    # buffer by 1 pixel
    geom = geom.buffer(1, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.bevel)
    # simplify with 1 pixel radius
    geom = geom.simplify(1)
    # intersect with image bounding box
    geom = geom.intersection(shapely.geometry.box(0, 0, shape_x, shape_y))
    # transform from pixel space into CRS space
    geom = shapely.affinity.affine_transform(
        geom,
        (
            grid.affine.a,
            grid.affine.b,
            grid.affine.d,
            grid.affine.e,
            grid.affine.xoff,
            grid.affine.yoff,
        ),
    )
    return geom


def create_thumbnail(
    rgb: tuple[Path, Path, Path],
    out: Path,
    out_scale=10,
    resampling=Resampling.average,
    static_stretch: tuple[int, int] = None,
    percentile_stretch: tuple[int, int] = (2, 98),
    compress_quality: int = 85,
    input_geobox: GeoBox = None,
):
    """
    Generate a thumbnail jpg image using the given three paths as red,green, blue.

    A linear stretch is performed on the colour. By default this is a dynamic 2% stretch
    (the 2% and 98% percentile values of the input). The static_stretch parameter will
    override this with a static range of values.

    If the input image has a valid no data value, the no data will
    be set to 0 in the output image.

    Any non-contiguous data across the colour domain, will be set to
    zero.
    """
    # No aux.xml file with our jpeg.
    with rasterio.Env(GDAL_PAM_ENABLED=False):
        with tempfile.TemporaryDirectory(dir=out.parent, prefix=".thumbgen-") as tmpdir:
            tmp_quicklook_path = Path(tmpdir) / "quicklook.tif"

            # We write an intensity-scaled, reprojected version of the dataset at full res.
            # Then write a scaled JPEG verison. (TODO: can we do it in one step?)
            ql_grid = _write_quicklook(
                rgb,
                tmp_quicklook_path,
                resampling,
                static_range=static_stretch,
                percentile_range=percentile_stretch,
                input_geobox=input_geobox,
            )
            out_crs = ql_grid.crs

            # Scale and write as JPEG to the output.
            (
                thumb_transform,
                thumb_width,
                thumb_height,
            ) = calculate_default_transform(
                out_crs,
                out_crs,
                ql_grid.shape[1],
                ql_grid.shape[0],
                *ql_grid.boundingbox,
                dst_width=ql_grid.shape[1] // out_scale,
                dst_height=ql_grid.shape[0] // out_scale,
            )
            thumb_args = dict(
                driver="JPEG",
                quality=compress_quality,
                height=thumb_height,
                width=thumb_width,
                count=3,
                dtype="uint8",
                nodata=0,
                transform=thumb_transform,
                crs=out_crs,
            )
            with rasterio.open(tmp_quicklook_path, "r") as ql_ds:
                ql_ds: DatasetReader
                with rasterio.open(out, "w", **thumb_args) as thumb_ds:
                    thumb_ds: DatasetWriter
                    for index in thumb_ds.indexes:
                        thumb_ds.write(
                            ql_ds.read(
                                index,
                                out_shape=(thumb_height, thumb_width),
                                resampling=resampling,
                            ),
                            index,
                        )


def create_thumbnail_from_numpy(
    rgb: tuple[numpy.array, numpy.array, numpy.array],
    out_scale=10,
    resampling=Resampling.average,
    static_stretch: tuple[int, int] = None,
    percentile_stretch: tuple[int, int] = (2, 98),
    compress_quality: int = 85,
    input_geobox: GeoBox = None,
    nodata: int = -999,
):
    """
    Generate a thumbnail as numpy arrays.

    Unlike the default `create_thumbnail` function, this is done entirely in-memory. It will likely require more
    memory but does not touch the filesystem.

    A linear stretch is performed on the colour. By default this is a dynamic 2% stretch
    (the 2% and 98% percentile values of the input). The static_stretch parameter will
    override this with a static range of values.

    Any non-contiguous data across the colour domain, will be set to zero.
    """
    ql_grid, numpy_array_list, ql_write_args = _write_to_numpy_array(
        rgb,
        resampling,
        static_range=static_stretch,
        percentile_range=percentile_stretch,
        input_geobox=input_geobox,
        nodata=nodata,
    )
    out_crs = ql_grid.crs

    # Scale and write as JPEG to the output.
    (
        thumb_transform,
        thumb_width,
        thumb_height,
    ) = calculate_default_transform(
        out_crs,
        out_crs,
        ql_grid.shape[1],
        ql_grid.shape[0],
        *ql_grid.boundingbox,
        dst_width=ql_grid.shape[1] // out_scale,
        dst_height=ql_grid.shape[0] // out_scale,
    )
    thumb_args = dict(
        driver="JPEG",
        quality=compress_quality,
        height=thumb_height,
        width=thumb_width,
        count=3,
        dtype="uint8",
        nodata=0,
        transform=thumb_transform,
        crs=out_crs,
    )

    with MemoryFile() as mem_tif_file:
        with mem_tif_file.open(**ql_write_args) as dataset:
            for i, data in enumerate(numpy_array_list):
                dataset.write(data, i + 1)

            with MemoryFile() as mem_jpg_file:
                with mem_jpg_file.open(**thumb_args) as thumbnail:
                    for index in thumbnail.indexes:
                        thumbnail.write(  # write the data from temp_tif to temp_jpg
                            dataset.read(
                                index,
                                out_shape=(thumb_height, thumb_width),
                                resampling=Resampling.average,
                            ),
                            index,
                        )

                return_bytes = mem_jpg_file.read()

    return return_bytes


def create_thumbnail_singleband(
    in_file: Path,
    out_file: Path,
    bit: int = None,
    lookup_table: dict[int, tuple[int, int, int]] = None,
):
    """
    Write out a JPG thumbnail from a singleband image.
    This takes in a path to a valid raster dataset and writes
    out a file with only the values of the bit (integer) as white
    """
    if bit is not None and lookup_table is not None:
        raise ValueError("Please set either bit or lookup_table, and not both of them")
    if bit is None and lookup_table is None:
        raise ValueError(
            "Please set either bit or lookup_table, you haven't set either of them"
        )

    with rasterio.open(in_file) as dataset:
        data = dataset.read()
        out_data, stretch = _filter_singleband_data(data, bit, lookup_table)

    meta = dataset.meta
    meta["driver"] = "GTiff"

    with tempfile.TemporaryDirectory() as temp_dir:
        if bit:
            # Only use one file, three times
            temp_file = Path(temp_dir) / "temp.tif"

            with rasterio.open(temp_file, "w", **meta) as tmpdataset:
                tmpdataset.write(out_data)
            create_thumbnail(
                (temp_file, temp_file, temp_file),
                out_file,
                static_stretch=stretch,
            )
        else:
            # Use three different files
            temp_files = tuple(Path(temp_dir) / f"temp_{i}.tif" for i in range(3))

            for i in range(3):
                with rasterio.open(temp_files[i], "w", **meta) as tmpdataset:
                    tmpdataset.write(out_data[i])
            create_thumbnail(temp_files, out_file, static_stretch=stretch)


def create_thumbnail_singleband_from_numpy(
    input_data: numpy.array,
    bit: int = None,
    lookup_table: dict[int, tuple[int, int, int]] = None,
    input_geobox: GeoBox = None,
    nodata: int = -999,
) -> bytes:
    """
    Output a thumbnail ready bytes from the input numpy array.
    This takes a valid raster data (numpy arrary) and return
    out bytes with only the values of the bit (integer) as white.
    """
    if bit is not None and lookup_table is not None:
        raise ValueError("Please set either bit or lookup_table, and not both of them")
    if bit is None and lookup_table is None:
        raise ValueError(
            "Please set either bit or lookup_table, you haven't set either of them"
        )

    out_data, stretch = _filter_singleband_data(input_data, bit, lookup_table)

    if bit:
        rgb = [out_data, out_data, out_data]
    else:
        rgb = out_data

    return create_thumbnail_from_numpy(
        rgb=rgb,
        static_stretch=stretch,
        input_geobox=input_geobox,
        nodata=nodata,
    )


def _filter_singleband_data(
    data: numpy.array,
    bit: int = None,
    lookup_table: dict[int, tuple[int, int, int]] = None,
):
    """
    Apply bit or lookup_table to filter the numpy array
    and generate the thumbnail content.
    """
    if bit is not None:
        out_data = numpy.copy(data)
        out_data[data != bit] = 0
        stretch = (0, bit)
    if lookup_table is not None:
        out_data = [
            numpy.full_like(data, 0),
            numpy.full_like(data, 0),
            numpy.full_like(data, 0),
        ]
        stretch = (0, 255)

        for value, rgb in lookup_table.items():
            for index in range(3):
                out_data[index][data == value] = rgb[index]
    return out_data, stretch


def _write_to_numpy_array(
    rgb: Sequence[numpy.array],
    resampling: Resampling,
    static_range: tuple[int, int],
    percentile_range: tuple[int, int] = (2, 98),
    input_geobox: GeoBox = None,
    nodata: int = -999,
) -> GeoBox:
    """
    Write an intensity-scaled wgs84 image using the given files as bands.
    """
    if input_geobox is None:
        raise NotImplementedError("generating geobox from numpy is't yet supported")

    out_crs = CRS(4326)
    (
        reprojected_transform,
        reprojected_width,
        reprojected_height,
    ) = calculate_default_transform(
        input_geobox.crs,
        out_crs,
        input_geobox.shape[1],
        input_geobox.shape[0],
        *input_geobox.boundingbox,
    )
    reproj_grid = GeoBox(
        (reprojected_height, reprojected_width), reprojected_transform, crs=out_crs
    )
    ql_write_args = dict(
        driver="GTiff",
        dtype="uint8",
        count=len(rgb),
        width=reproj_grid.shape[1],
        height=reproj_grid.shape[0],
        transform=reproj_grid.affine,
        crs=reproj_grid.crs,
        nodata=0,
        tiled="yes",
    )

    # Only set blocksize on larger imagery; enables reduced resolution processing
    if reproj_grid.shape[0] > 512:
        ql_write_args["blockysize"] = 512
    if reproj_grid.shape[1] > 512:
        ql_write_args["blockxsize"] = 512

    # Calculate combined nodata mask
    valid_data_mask = numpy.ones(input_geobox.shape, dtype="bool")
    calculated_range = read_valid_mask_and_value_range(
        valid_data_mask, _iter_arrays(rgb, nodata=nodata), percentile_range
    )

    output_list = []

    for band_no, (image, nodata) in enumerate(
        _iter_arrays(rgb, nodata=nodata), start=1
    ):
        reprojected_data = numpy.zeros(reproj_grid.shape, dtype=numpy.uint8)
        reproject(
            rescale_intensity(
                image,
                image_null_mask=~valid_data_mask,
                in_range=(static_range or calculated_range),
                out_range=(1, 255),
                out_dtype=numpy.uint8,
            ),
            reprojected_data,
            src_crs=input_geobox.crs,
            src_transform=input_geobox.affine,
            src_nodata=0,
            dst_crs=reproj_grid.crs,
            dst_nodata=0,
            dst_transform=reproj_grid.affine,
            resampling=resampling,
            num_threads=2,
        )
        output_list.append(reprojected_data)
        del reprojected_data

    return reproj_grid, output_list, ql_write_args


def _write_quicklook(
    rgb: Sequence[Path],
    dest_path: Path,
    resampling: Resampling,
    static_range: tuple[int, int],
    percentile_range: tuple[int, int] = (2, 98),
    input_geobox: GeoBox = None,
) -> GeoBox:
    """
    Write an intensity-scaled wgs84 image using the given files as bands.
    """
    if input_geobox is None:
        with rasterio.open(rgb[0]) as ds:
            input_geobox = GeoBox.from_rio(ds)

    out_crs = CRS(4326)
    (
        reprojected_transform,
        reprojected_width,
        reprojected_height,
    ) = calculate_default_transform(
        input_geobox.crs,
        out_crs,
        input_geobox.shape[1],
        input_geobox.shape[0],
        *input_geobox.boundingbox,
    )
    reproj_grid = GeoBox(
        (reprojected_height, reprojected_width), reprojected_transform, crs=out_crs
    )
    ql_write_args = dict(
        driver="GTiff",
        dtype="uint8",
        count=len(rgb),
        width=reproj_grid.shape[1],
        height=reproj_grid.shape[0],
        transform=reproj_grid.affine,
        crs=reproj_grid.crs,
        nodata=0,
        tiled="yes",
    )

    # Only set blocksize on larger imagery; enables reduced resolution processing
    if reproj_grid.shape[0] > 512:
        ql_write_args["blockysize"] = 512
    if reproj_grid.shape[1] > 512:
        ql_write_args["blockxsize"] = 512

    with rasterio.open(dest_path, "w", **ql_write_args) as ql_ds:
        ql_ds: DatasetWriter

        # Calculate combined nodata mask
        valid_data_mask = numpy.ones(input_geobox.shape, dtype="bool")
        calculated_range = read_valid_mask_and_value_range(
            valid_data_mask, _iter_images(rgb), percentile_range
        )

        for band_no, (image, nodata) in enumerate(_iter_images(rgb), start=1):
            reprojected_data = numpy.zeros(reproj_grid.shape, dtype=numpy.uint8)
            reproject(
                rescale_intensity(
                    image,
                    image_null_mask=~valid_data_mask,
                    in_range=(static_range or calculated_range),
                    out_range=(1, 255),
                    out_dtype=numpy.uint8,
                ),
                reprojected_data,
                src_crs=input_geobox.crs,
                src_transform=input_geobox.affine,
                src_nodata=0,
                dst_crs=reproj_grid.crs,
                dst_nodata=0,
                dst_transform=reproj_grid.affine,
                resampling=resampling,
                num_threads=2,
            )
            ql_ds.write(reprojected_data, band_no)
            del reprojected_data

    return reproj_grid


LazyImages = Iterable[tuple[numpy.ndarray, int]]


def _iter_images(rgb: Sequence[Path]) -> LazyImages:
    """
    Lazily load a series of single-band images from a path.

    Yields the image array and nodata value.
    """
    for path in rgb:
        with rasterio.open(path) as ds:
            ds: DatasetReader
            if ds.count != 1:
                raise NotImplementedError(
                    "multi-band measurement files aren't yet supported"
                )
            yield ds.read(1), ds.nodata


def _iter_arrays(rgb: Sequence[numpy.array], nodata: int) -> LazyImages:
    """
    Lazily load a series of single-band images from a path.

    Yields the image array and nodata value.
    """
    for data in rgb:
        yield data, nodata


def read_valid_mask_and_value_range(
    valid_data_mask: numpy.ndarray,
    images: LazyImages,
    calculate_percentiles: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """
    Read the given images, filling in a valid data mask and optional pixel percentiles.
    """
    calculated_range = (-sys.maxsize - 1, sys.maxsize)
    for array, nodata in images:
        valid_data_mask &= array != nodata

        if calculate_percentiles is not None:
            the_data = array[valid_data_mask]
            # Check if there's a non-empty array first
            if the_data.any():
                # Numpy changed the 'interpolation' method, but we need to still support the
                # older Python 3.6 module at NCI.
                if numpy.__version__ < "1.22":
                    low, high = numpy.percentile(
                        the_data, calculate_percentiles, interpolation="nearest"
                    )
                else:
                    low, high = numpy.percentile(
                        the_data, calculate_percentiles, method="nearest"
                    )
                calculated_range = (
                    max(low, calculated_range[0]),
                    min(high, calculated_range[1]),
                )

    return calculated_range


def rescale_intensity(
    image: numpy.ndarray,
    in_range: tuple[int, int],
    out_range: tuple[int, int] | None = None,
    image_nodata: int = None,
    image_null_mask: numpy.ndarray = None,
    out_dtype=numpy.uint8,
    out_nodata=0,
) -> numpy.ndarray:
    """
    Based on scikit-image's rescale_intensity, but does fewer copies/allocations of the array.

    (and it saves us bringing in the entire dependency for one small method)
    """
    if image_null_mask is None:
        if image_nodata is None:
            raise ValueError("Must specify either a null mask or a nodata val")
        image_null_mask = image == image_nodata

    imin, imax = in_range
    omin, omax = out_range or (numpy.iinfo(out_dtype).min, numpy.iinfo(out_dtype).max)

    # The intermediate calculation will need floats.
    # We'll convert to it immediately to avoid modifying the input array
    image = image.astype(numpy.float64)

    numpy.clip(image, imin, imax, out=image)
    image -= imin
    image /= float(imax - imin)
    image *= omax - omin
    image += omin
    image = image.astype(out_dtype)
    image[image_null_mask] = out_nodata
    return image
