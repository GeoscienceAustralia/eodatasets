import tempfile
from pathlib import Path

import rasterio
from odc.geo.geobox import GeoBox

from eodatasets3.images import (
    create_thumbnail_singleband,
    create_thumbnail_singleband_from_numpy,
)

from . import assert_image


def test_thumbnail_bitflag(input_uint8_tif: Path):
    outfile = Path(tempfile.gettempdir()) / "test-bitflag.jpg"

    water = 128

    create_thumbnail_singleband(input_uint8_tif, Path(outfile), bit=water)

    assert_image(outfile, bands=3)


def test_thumbnail_lookuptable(input_uint8_tif_2: Path):
    outfile = Path(tempfile.gettempdir()) / "test-lookuptable.jpg"

    wofs_lookup = {
        0: [150, 150, 110],  # dry
        1: [255, 255, 255],  # nodata,
        16: [119, 104, 87],  # terrain
        32: [89, 88, 86],  # cloud_shadow
        64: [216, 215, 214],  # cloud
        80: [242, 220, 180],  # cloudy terrain
        128: [79, 129, 189],  # water
        160: [51, 82, 119],  # shady water
        192: [186, 211, 242],  # cloudy water
    }

    create_thumbnail_singleband(
        input_uint8_tif_2, Path(outfile), lookup_table=wofs_lookup
    )

    assert_image(outfile, bands=3)


def test_thumbnail_from_numpy_bitflag(input_uint8_tif: Path):
    outfile = Path(tempfile.gettempdir()) / "test-bitflag.jpg"
    water = 128

    with rasterio.open(input_uint8_tif) as ds:
        input_geobox = GeoBox.from_rio(ds)
        data = ds.read(1)

        image_bytes = create_thumbnail_singleband_from_numpy(
            input_data=data, input_geobox=input_geobox, bit=water
        )

        with open(outfile, "wb") as jpeg_file:
            jpeg_file.write(image_bytes)

        assert_image(outfile, bands=3)


def test_thumbnail_from_numpy_lookuptable(input_uint8_tif_2: Path):
    outfile = Path(tempfile.gettempdir()) / "test-lookuptable.jpg"
    wofs_lookup = {
        0: [150, 150, 110],  # dry
        1: [255, 255, 255],  # nodata,
        16: [119, 104, 87],  # terrain
        32: [89, 88, 86],  # cloud_shadow
        64: [216, 215, 214],  # cloud
        80: [242, 220, 180],  # cloudy terrain
        128: [79, 129, 189],  # water
        160: [51, 82, 119],  # shady water
        192: [186, 211, 242],  # cloudy water
    }

    with rasterio.open(input_uint8_tif_2) as ds:
        input_geobox = GeoBox.from_rio(ds)
        data = ds.read(1)

        image_bytes = create_thumbnail_singleband_from_numpy(
            input_data=data, input_geobox=input_geobox, lookup_table=wofs_lookup
        )

        with open(outfile, "wb") as jpeg_file:
            jpeg_file.write(image_bytes)

        assert_image(outfile, bands=3)
