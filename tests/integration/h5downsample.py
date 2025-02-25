"""
Alter a HDF5 file, making it much smaller by shrinking all embedded images.

It modifies the file in-place!

(Intended for creating test datasets where the pixels don't
matter much but the structure does. The downsampling is dirty.)
"""

import re
import shutil
from pathlib import Path

import click
import h5py
from click import secho, style

from eodatasets3.ui import PathPath
from eodatasets3.wagl import find_a_granule_name


def find_h5_paths(h5_obj: h5py.Group, dataset_class: str = "") -> list[str]:
    """
    Find all objects in a h5 of the given class, returning their path.

    (class examples: IMAGE, TABLE. SCALAR)
    """
    items = []

    def _find(name, obj):
        if obj.attrs.get("CLASS") == dataset_class:
            items.append(name)

    h5_obj.visititems(_find)
    return items


RES_GROUP_PATH = re.compile(r"(.*/RES-GROUP-\d+)/")


@click.command(help=__doc__)
@click.argument("input_path", type=PathPath(dir_okay=False, readable=True))
@click.option("--factor", type=int, default=100)
@click.option("--anti-alias/--no-anti-alias", is_flag=True, default=False)
def downsample(input_path: Path, factor: int, anti_alias: bool):
    # Fail early if h5repack cli command is not available.
    from sh import gdal_translate, h5repack

    granule_name = find_a_granule_name(input_path)
    fmask_image = input_path.with_name(f"{granule_name}.fmask.img")

    nbar_size = None

    # Create temporary directory
    original = input_path.with_suffix(".original.h5")
    secho(f"Creating backup to {original.name}")
    shutil.copy(input_path, original)

    try:
        with h5py.File(input_path, "r+") as f:
            image_paths = find_h5_paths(f, "IMAGE")
            secho(f"Found {len(image_paths)} images")
            for i, image_path in enumerate(image_paths):
                old_image: h5py.Dataset | None = f[image_path]

                def info(msg: str):
                    secho(
                        f"{i: 4}/{len(image_paths)} {style(repr(image_path), fg='blue')}: {msg}"
                    )

                old_shape = old_image.shape
                if all(dim_size < factor for dim_size in old_shape):
                    info("Skipping")
                    continue

                attrs = dict(old_image.attrs.items())
                old_geotransform = attrs["geotransform"]

                new_data = old_image[()][::factor, ::factor]
                new_shape = new_data.shape
                info(f"New shape: {new_shape!r}")
                del old_image
                del f[str(image_path)]

                folder, name = image_path.rsplit("/", 1)
                parent: h5py.Group = f[str(folder)]

                image = parent.create_dataset(name, new_shape, data=new_data)
                new_geotransform = list(old_geotransform)
                new_geotransform[1] *= old_shape[1] / new_shape[1]
                new_geotransform[5] *= old_shape[0] / new_shape[0]
                attrs["geotransform"] = new_geotransform
                image.attrs.update(attrs)

                # Update any res group with the new resolution.
                res_group_path = _get_res_group_path(image_path)
                if res_group_path:
                    res_group = f[res_group_path]
                    res_group.attrs["resolution"] = [
                        abs(new_geotransform[5]),
                        abs(new_geotransform[1]),
                    ]

                if "/NBAR/" in image_path:
                    nbar_size = new_shape

        if nbar_size is None:
            raise ValueError("No nbar image found?")

        # We need to repack the file to actually free up the space.
        repacked = input_path.with_suffix(".repacked.h5")
        h5repack("-f", "GZIP=5", input_path, repacked)
        repacked.rename(input_path)
    except Exception:
        secho("Restoring backup")
        original.rename(input_path)
        raise

    if fmask_image.exists():
        original = fmask_image.with_suffix(f".original{fmask_image.suffix}")
        secho(f"Creating fmask backup to {original.name}")
        shutil.copy(fmask_image, original)
        secho(f"Scaling fmask {fmask_image}")

        tmp = fmask_image.with_suffix(f".tmp.{fmask_image.suffix}")
        gdal_translate("-outsize", nbar_size[1], nbar_size[0], fmask_image, tmp)
        tmp.rename(fmask_image)


def _get_res_group_path(image_path: str) -> str | None:
    """
    >>> _get_res_group_path('LC80920842016180LGN01/RES-GROUP-1/STANDARDISED-PRODUCTS/REFLECTANCE/NBART/BAND-7')
    'LC80920842016180LGN01/RES-GROUP-1'
    >>> # Nothing if not in a res group.
    >>> _get_res_group_path('LC80920842016180LGN01/SATELLITE-SOLAR/SOLAR-ZENITH')
    """
    m = RES_GROUP_PATH.match(image_path)
    if m:
        return m.group(1)
    return None


if __name__ == "__main__":
    downsample()
