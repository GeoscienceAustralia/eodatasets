"""
Create index files from the command-line
"""

import datetime
import urllib.parse
import uuid
from collections.abc import Iterable
from pathlib import Path

import click
import rasterio
import rasterio.crs
from rasterio.io import DatasetReader

from eodatasets3 import serialise
from eodatasets3.utils import ItemProvider, read_paths_from_file

from ..metadata.valid_region import valid_region

NOAA_WATER_VAPOUR_NS = uuid.UUID(hex="857bd048-8c86-4670-a2b4-5dbea26d7692")


def get_uuid(collection: DatasetReader, idx: int):
    """
    Returns a deterministic uuid based on band index and gdal checksum
    """
    origin = collection.tags()["NC_GLOBAL#References"]

    return uuid.uuid5(
        NOAA_WATER_VAPOUR_NS,
        "{}?{}".format(
            origin,
            urllib.parse.urlencode(
                {
                    "checksum": collection.checksum(idx),
                    "band_index": idx,
                    "filename": Path(collection.name).stem,
                }
            ),
        ),
    )


def process_datasets(dataset: Path) -> Iterable[dict]:
    """
    Generates a metadata document for each band available
    in the water vapour file.

    Each band is treated as a separate dataset since the
    source file is updated in-place; allowing the file
    to be modified but a static reference is provided for
    each observation

    """
    datasets = []
    creation_dt = datetime.datetime.now(datetime.UTC)
    geometry = valid_region([str(dataset)])

    with rasterio.open(str(dataset), "r") as collection:
        collection_start_date = datetime.datetime.strptime(
            collection.tags()["time#units"], "hours since %Y-%m-%d %H:%M:%S.%f"
        )

        for _idx in collection.indexes:
            time_in_hours = int(collection.tags(_idx)["NETCDF_DIM_time"])
            ds_dt = (
                collection_start_date + datetime.timedelta(hours=time_in_hours)
            ).replace(tzinfo=datetime.timezone.utc)

            md = {}
            md["id"] = str(get_uuid(collection, _idx))
            md["product"] = {
                "href": "https://collections.dea.ga.gov.au/noaa_c_c_prwtreatm_1"
            }
            md["crs"] = "epsg:4236"
            md["datetime"] = ds_dt.isoformat()
            md["geometry"] = geometry
            md["grids"] = {"default": {}}
            md["grids"]["default"]["shape"] = list(collection.shape)
            md["grids"]["default"]["transform"] = list(collection.transform)
            md["lineage"] = {}
            md["measurements"] = {
                "water_vapour": {"band": _idx, "layer": "pr_wtr", "path": dataset.name}
            }
            md["properties"] = {
                "item:providers": [
                    {
                        "name": "NOAA/OAR/ESRL PSD",
                        "roles": [ItemProvider.PRODUCER.value],
                        "url": "https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.derived.surface.html",
                    }
                ],
                "odc:creation_datetime": creation_dt.isoformat(),
                "odc:file_format": "NetCDF",
            }

            datasets.append(md)

    return datasets


def _process_datasets(output_dir, datasets):
    """
    Wrapper function for processing multiple datasets
    """
    for dataset in datasets:
        docs = process_datasets(dataset)
        outfile = output_dir / (dataset.stem + ".ga-md.yaml")
        serialise.dump_yaml(outfile, *docs)


@click.command(
    help="""\b
        Prepare NCEP/NCAR reanalysis 1 water pressure datasets for indexing into a Data Cube.
        This prepare scripts supports netCDF water pressure files only.

        Example usage: yourscript.py --output [directory] input_file1 input_file2"""
)
@click.option(
    "--output",
    "output_dir",
    help="Write datasets into this directory",
    type=click.Path(exists=False, writable=True, dir_okay=True),
)
@click.argument(
    "datasets", type=click.Path(exists=True, readable=True, writable=False), nargs=-1
)
@click.option(
    "-f",
    "dataset_listing_files",
    type=click.Path(exists=True, readable=True, writable=False),
    help="file containing a list of input paths (one per line)",
    multiple=True,
)
def main(output_dir, datasets, dataset_listing_files):
    datasets = [Path(p) for p in datasets]
    for listing_file in dataset_listing_files:
        datasets.extend(read_paths_from_file(Path(listing_file)))

    return _process_datasets(Path(output_dir), datasets)


if __name__ == "__main__":
    main()
