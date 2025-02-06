"""
Prepare eo3 metadata for Sentinel-2 Level 1C data produced by Sinergise or ESA.

Takes ESA zipped datasets or Sinergise dataset directories
"""

import fnmatch
import json
import os.path
import re
import sys
import traceback
import uuid
import zipfile
from collections.abc import Iterable, Mapping
from multiprocessing import Pool
from pathlib import Path

import click
import structlog
from attr import define
from datacube.cfg import ODCConfig
from datacube.index import index_connect
from datacube.model import Dataset
from datacube.ui.click import config_option, environment_option
from datacube.utils.uris import normalise_path
from defusedxml import minidom

from eodatasets3 import DatasetDoc, DatasetPrepare, names, serialise
from eodatasets3.prepare.s2_common import FolderInfo, RegionLookup
from eodatasets3.properties import Eo3Interface
from eodatasets3.ui import PathPath
from eodatasets3.utils import pass_config

# Static namespace to generate uuids for datacube indexing
S2_UUID_NAMESPACE = uuid.UUID("9df23adf-fc90-4ec7-9299-57bd536c7590")

_LOG = structlog.get_logger()

SENTINEL_MSI_BAND_ALIASES = {
    "01": "coastal_aerosol",
    "02": "blue",
    "03": "green",
    "04": "red",
    "05": "red_edge_1",
    "06": "red_edge_2",
    "07": "red_edge_3",
    "08": "nir_1",
    "8A": "nir_2",
    "09": "water_vapour",
    "10": "swir_1_cirrus",
    "11": "swir_2",
    "12": "swir_3",
}


def process_sinergise_product_info(product_path: Path) -> dict:
    with product_path.open() as fp:
        product = json.load(fp)

    if len(product["tiles"]) > 1:
        raise NotImplementedError("No support for multi-tiled products yet")
    tile = product["tiles"][0]

    utm_zone = tile["utmZone"]
    latitude_band = tile["latitudeBand"]
    grid_square = tile["gridSquare"]
    return {
        "sentinel:product_name": product["name"],
        "sinergise_product_id": product["id"],
        "odc:region_code": f"{utm_zone}{latitude_band}{grid_square}",
        "sentinel:utm_zone": utm_zone,
        "sentinel:latitude_band": latitude_band,
        "sentinel:grid_square": grid_square,
    }


def _node_name(el):
    """Get a human-readable path to the given xml node"""
    branch = [el]
    while branch[-1].parentNode is not None:
        branch.append(branch[-1].parentNode)

    branch.reverse()
    return "/".join(n.localName for n in branch[1:])


def _value(root, *tags: str, type_=None):
    """Get the text contents of a tag in an xml document.

    Takes the tag name(s) to search for.
    """
    el = root

    for tag in tags:
        found = el.getElementsByTagName(tag)
        if not found:
            raise ValueError(f"Element not found in document: {tags!r}")
        if len(found) > 1:
            vs = "\n\t".join([_node_name(e) for e in found])
            raise ValueError(f"Multiple matches found for tag: {tags!r}: \n\t{vs}")

        el = found[0]

    el.normalize()
    if not len(el.childNodes) == 1:
        raise NotImplementedError(
            f"Not yet supported: Multiple child tags found for {tags!r}"
        )
    value = el.firstChild.data.strip()

    if not value:
        raise ValueError(f"Empty value for document at {tags!r}")

    if type_ is not None:
        value = type_(value)
    return value


def process_tile_metadata(contents: str) -> dict:
    """
    Tile xml metadata format, as described by
    xmlns https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd
    """
    root = minidom.parseString(contents)

    resolution = min(
        int(i.attributes["resolution"].value) for i in root.getElementsByTagName("Size")
    )
    tile_id = _value(root, "TILE_ID")

    region_code = tile_id.split("_")[-2]
    if not region_code.startswith("T"):
        raise RuntimeError(
            f"Tile id is not recognised -- not a region code? Please report this. {tile_id!r}"
        )
    region_code = region_code[1:]

    return {
        "datetime": _value(root, "SENSING_TIME"),
        "eo:cloud_cover": _value(root, "CLOUDY_PIXEL_PERCENTAGE", type_=float),
        "eo:gsd": resolution,
        "eo:sun_azimuth": _value(root, "Mean_Sun_Angle", "AZIMUTH_ANGLE", type_=float),
        "eo:sun_elevation": _value(root, "Mean_Sun_Angle", "ZENITH_ANGLE", type_=float),
        "odc:processing_datetime": _value(root, "ARCHIVING_TIME"),
        "sentinel:datastrip_id": _value(root, "DATASTRIP_ID"),
        "sentinel:sentinel_tile_id": tile_id,
        "odc:region_code": region_code,
    }


def process_datastrip_metadata(contents: str) -> list[str] | dict:
    """
    Datastrip metadata format, as described by
    xmlns https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Datastrip_Metadata.xsd

    Returns a list of tile names in this datastrip, and the properties that could be extracted.
    """
    root = minidom.parseString(contents)

    resolution_tags = root.getElementsByTagName("RESOLUTION")
    if resolution_tags:
        resolution = min(int(i.firstChild.data) for i in resolution_tags)
    elif root.getElementsByTagName("SPACECRAFT_NAME")[0].firstChild.data.startswith(
        "Sentinel-2"
    ):
        resolution = 10.0
    else:
        raise ValueError("No resolution in datastrip metadata and unknown craft")

    tile_ids = [
        tile_tag.attributes["tileId"] for tile_tag in root.getElementsByTagName("Tile")
    ]

    return tile_ids, {
        "sentinel:reception_station": _value(root, "RECEPTION_STATION"),
        "sentinel:processing_center": _value(root, "PROCESSING_CENTER"),
        "eo:gsd": resolution,
    }


def process_user_product_metadata(contents: str, filename_stem: str = None) -> dict:
    root = minidom.parseString(contents)

    # - On newer datasets, get the product URI from metadata.
    # - On older datasets, get it from the filename.
    # (the metadata field is useless on older datasets, the filename is useless/fixed on newer datasets)
    raw_product_uri = _value(root, "PRODUCT_URI")

    if raw_product_uri.startswith("S2"):
        product_uri = raw_product_uri.split(".")[0]
    else:
        product_uri = filename_stem

    return {
        "eo:platform": _value(root, "SPACECRAFT_NAME"),
        "sat:relative_orbit": _value(root, "SENSING_ORBIT_NUMBER", type_=int),
        "sat:orbit_state": _value(root, "SENSING_ORBIT_DIRECTION").lower(),
        "sentinel:datatake_type": _value(root, "DATATAKE_TYPE"),
        "sentinel:processing_baseline": _value(root, "PROCESSING_BASELINE"),
        "sentinel:product_name": product_uri,
        "eo:cloud_cover": _value(root, "Cloud_Coverage_Assessment"),
    }


def _get_stable_id(p: Eo3Interface) -> uuid.UUID:
    # These should have been extracted from our metadata files!
    producer = p.producer
    product_name = p.properties["sentinel:product_name"]
    tile_id = p.properties["sentinel:sentinel_tile_id"]
    return uuid.uuid5(
        S2_UUID_NAMESPACE,
        ":".join((producer, product_name, tile_id)),
    )


def list_granules(dataset_location: Path) -> list[str] | None:
    """
    Get a list of granule ids if it's a dataset that may contain multiple.
    """
    if dataset_location.suffix.lower() == ".json":
        # Sinergise
        return None

    # Each granule is a directory. Get a unique list of all the granule directory names (ie. IDs).
    granules = set()

    # We only want granule directories containing imagery files.
    # Some "null" datasets have been found with empty folders.
    granule_folder_pattern = re.compile(r"/GRANULE/([^/]+)/.*jp2$")

    with zipfile.ZipFile(dataset_location, "r") as z:
        for name in z.namelist():
            match = granule_folder_pattern.search(name)
            if match:
                granules.add(match.group(1))

    return sorted(granules)


def prepare_and_write(
    dataset_location: Path,
    output_yaml: Path,
    producer: str,
    granule_id: str = None,
    embed_location: bool = None,
) -> tuple[DatasetDoc, Path]:
    if embed_location is None:
        # Default to embedding the location if they're not in the same folder.
        embed_location = output_yaml.parent not in dataset_location.parents
        _LOG.debug(
            "Auto-embed location?",
            auto_embed=bool(embed_location),
            data_location=dataset_location.parent,
            yaml_location=output_yaml.parent,
        )

    with DatasetPrepare(
        metadata_path=output_yaml,
        dataset_location=dataset_location,
    ) as p:
        p.properties["odc:producer"] = producer

        if producer == "esa.int":
            jp2_offsets = _extract_esa_fields(
                dataset_location, p, granule_id=granule_id
            )
        elif producer == "sinergise.com":
            jp2_offsets = _extract_sinergise_fields(dataset_location.parent, p)
        else:
            raise NotImplementedError(
                f"Unknown s2 producer {producer}. Expected 'sinergise.com' or 'esa.int'"
            )

        p.dataset_id = _get_stable_id(p)

        p.platform = _get_platform_name(p.properties)
        p.instrument = "MSI"
        p.constellation = "sentinel-2"

        # TODO: How to read collection number from metadata? (once ESA etc add one)
        collection_number = 0
        p.dataset_version = f"{collection_number}.0.{p.processed:%Y%m%d}"

        p.properties["odc:file_format"] = "JPEG2000"
        p.product_family = "level1"

        for path in jp2_offsets:
            band_number = _extract_band_number(path.stem)
            if band_number.lower() in ("tci", "pvi", "preview"):
                continue
            if band_number not in SENTINEL_MSI_BAND_ALIASES:
                raise RuntimeError(
                    f"Unknown band number {band_number!r} in image {path}"
                )

            p.note_measurement(
                path=path,
                name=SENTINEL_MSI_BAND_ALIASES[band_number],
                relative_to_dataset_location=True,
            )

        dataset_id, metadata_path = p.done(embed_location=embed_location)
        doc = serialise.from_doc(p.written_dataset_doc, skip_validation=True)
        if not doc.locations:
            doc.locations = [names.resolve_location(dataset_location)]
        return doc, metadata_path


def _get_platform_name(p: Mapping) -> str:
    # Grab it from the datastrip id,
    # Eg. S2B_OPER_MSI_L1C_DS_VGS4_20210426T010904_S20210425T235239_N03

    datastrip_id: str = p.get("sentinel:datastrip_id")
    if not datastrip_id:
        raise ValueError(
            "Could not find a sentinel datastrip ID after reading all metadata documents"
        )
    if not datastrip_id.lower().startswith("s2"):
        raise RuntimeError("Expected sentinel datastrip-id to start with 's2'!")
    platform_variant = datastrip_id[1:3].lower()
    platform_name = f"sentinel-{platform_variant}"
    return platform_name


def _extract_sinergise_fields(path: Path, p: DatasetPrepare) -> Iterable[Path]:
    """Extract Sinergise metadata and return list of image offsets"""
    product_info_path = path / "productInfo.json"
    metadata_xml_path = path / "metadata.xml"

    if not product_info_path.exists():
        raise ValueError(
            "No productInfo.json file found. "
            "Are you sure the input is a sinergise dataset folder?"
        )

    # Tile/Granule metadata
    p.properties.update(process_tile_metadata(metadata_xml_path.read_text()))
    p.note_accessory_file("metadata:s2_tile", metadata_xml_path)

    # Whole-product metadata
    for prop, value in process_sinergise_product_info(product_info_path).items():
        # We don't want to override properties that came from the (more-specific) tile metadata.
        if prop not in p.properties:
            p.properties[prop] = value
    p.note_accessory_file("metadata:sinergise_product_info", product_info_path)

    # TODO: sinergise folders could `process_datastrip_metadata()` in an outer directory?

    return list(path.glob("*.jp2"))


def _extract_esa_fields(
    dataset: Path, p: DatasetPrepare, granule_id: str
) -> Iterable[Path]:
    """Extract ESA metadata and return list of image offsets"""
    with zipfile.ZipFile(dataset, "r") as z:

        def find(*patterns) -> Iterable[str]:
            internal_folder_name = os.path.commonprefix(z.namelist())
            for s in z.namelist():
                if any(
                    re.match(pattern, s[len(internal_folder_name) :])
                    for pattern in patterns
                ):
                    yield s

        def one(*patterns) -> str:
            """Find one path ending in the given name"""

            [*matches] = find(*patterns)
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one file matching {patterns}, but found {len(matches)} of them in {dataset}:"
                    f"\n\t{matches}"
                )
            return matches[0]

        # If multiple datasetrips, find the one with this granule id?
        datastrip_files = find(r".*MTD_DS\.xml$", r"DATASTRIP/S2[^/]+/S2[^/]+\.xml$")
        if not datastrip_files:
            raise ValueError(f"No datastrip metadatas found in input? {dataset}")

        datastrip_md = None
        for datastrip_file in datastrip_files:
            inner_tiles, datastrip_md = process_datastrip_metadata(
                z.read(datastrip_file).decode("utf-8")
            )

            if granule_id in inner_tiles:
                # We found the datastrip for this tile.
                break

        if not datastrip_md:
            raise ValueError(
                f"None of the found datastrip metadatas cover this granule id {granule_id}"
            )

        p.properties.update(datastrip_md)
        p.note_accessory_file("metadata:s2_datastrip", datastrip_file)

        # Get the specific granule metadata
        [*tile_mds] = find(
            r".*MTD_TL\.xml$",
            rf"GRANULE/{granule_id}/S2.*\.xml" if granule_id else r"GRANULE/S2.*\.xml",
        )
        if not tile_mds:
            raise ValueError(
                "Could not find any tile metadata files in the dataset. "
                + (
                    f"(Searching for granule {granule_id!r})"
                    if granule_id
                    else "(Any granule)"
                )
            )
        if len(tile_mds) != 1:
            raise ValueError(
                f"Expected exactly one granule in package, since no granule id was provided: {dataset}:"
                f"\n\t{tile_mds}"
            )
        tile_md = tile_mds[0]

        p.properties.update(process_tile_metadata(z.read(tile_md).decode("utf-8")))
        p.note_accessory_file("metadata:s2_tile", tile_md)

        # Wider product metadata.
        user_product_md = one(r".*MTD_MSIL1C\.xml", r"S2.*\.xml")
        for prop, value in process_user_product_metadata(
            z.read(user_product_md).decode("utf-8"),
            filename_stem=Path(user_product_md).stem,
        ).items():
            # We don't want to override properties that came from the (more-specific) tile metadata.
            if prop not in p.properties:
                p.properties[prop] = value
        p.note_accessory_file("metadata:s2_user_product", user_product_md)

        # Get the list of images in the same tile folder.
        img_folder_offset = (Path(tile_md).parent / "IMG_DATA").as_posix()
        return [
            Path(p)
            for p in z.namelist()
            if p.startswith(img_folder_offset) and p.endswith(".jp2")
        ]


def _extract_band_number(name: str) -> str:
    """
    >>> _extract_band_number('B03')
    '03'
    >>> _extract_band_number('T55HFA_20201011T000249_B01')
    '01'
    """
    return name.split("_")[-1].replace("B", "")


def _rglob_with_self(path: Path, pattern: str) -> Iterable[Path]:
    if fnmatch.fnmatch(path.name, pattern):
        yield path
        return
    yield from path.rglob(pattern)


class YearMonth(click.ParamType):
    """A YYYY-MM string converted to an integer tuple"""

    name = "year-month"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        hp = value.split("-")
        if len(hp) != 2:
            self.fail("Expect value in YYYY-MM format")

        year, month = hp
        if not year.isdigit() or not month.isdigit():
            self.fail("Expect value in YYYY-MM format")

        return int(year), int(month)


@define(order=True, hash=True)
class Job:
    """A dataset to process"""

    dataset_path: Path
    output_yaml_path: Path
    granule_id: str | None

    # "sinergise.com" / "esa.int"
    producer: str
    embed_location: bool


@define(order=True, hash=True)
class InputDataset:
    name: str
    producer: str

    path: Path
    base_folder: Path

    @property
    def metadata(self) -> FolderInfo | None:
        return FolderInfo.for_path(self.path)

    @property
    def granule_offsets(self) -> list[str] | None:
        """
        Get the list of granule offsets in the dataset.

        Offsets are the subdirectory names used inside ESA dataset packages.

        Example: 'L1C_T55HFA_A018789_20201011T000244'
        """
        return list_granules(self.path)


REGION_CODE_FORMAT = re.compile(
    r"T"
    # A landsat path/row
    r"([0-9]{6}"
    # ... or a sentinel MGRS
    r"|[0-9A-Z]{5})"
)


def get_region_code_from_granule_offset(granule_id: str) -> str:
    """
    >>> get_region_code_from_granule_offset('L1C_T55HFA_A018789_20201011T000244')
    '55HFA'
    >>> get_region_code_from_granule_offset('S2A_OPER_MSI_L1C_TL_EPA__20161001T102755_A000900_T53KNU_N02.04')
    '53KNU'
    """
    for section in granule_id.split("_"):
        match = REGION_CODE_FORMAT.match(section)
        if match:
            return match.group(1)

    raise ValueError(f"Granule offset is in an unexpected form? {granule_id!r}")


@click.command(help=__doc__)
@click.option("-v", "--verbose", is_flag=True)
@click.argument(
    "datasets",
    type=PathPath(),
    nargs=-1,
)
@click.option(
    "-f",
    "--datasets-path",
    help="A file to read input dataset paths from, one per line",
    required=False,
    type=PathPath(
        exists=True, readable=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
)
@click.option(
    "-j",
    "--jobs",
    "workers",
    help="Number of workers to run in parallel",
    type=int,
    default=1,
)
@click.option(
    "--overwrite-existing/--skip-existing",
    is_flag=True,
    default=False,
    help="Overwrite if exists (otherwise skip)",
)
@click.option(
    "--embed-location/--no-embed-location",
    is_flag=True,
    default=None,
    help="Embed the location of the dataset in the metadata? "
    "(if you wish to store them separately. default: auto)",
)
@click.option(
    "--always-granule-id/--never-granule-id",
    "always_granule_id",
    is_flag=True,
    default=None,
    help="Include the granule id in metadata filenames? (default: auto -- include only for multi-granule files). "
    "Beware that multi-granule datasets without a granule id in the filename will overwrite each-other",
)
@click.option(
    "--throughly-check-existing/--cheaply-check-existing",
    "thoroughly_check_existing",
    is_flag=True,
    default=False,
    help="Should we open every dataset to check if *all* inner granules have been produced? Default: false.",
)
@click.option(
    "--provider",
    default=None,
    type=click.Choice(
        [
            "sinergise.com",
            "esa.int",
        ]
    ),
    help="Restrict scanning to only packages of the given provider. "
    "(ESA assumes a zip file, sinergise a directory)",
)
@click.option(
    "--output-base",
    help="Write metadata files into a directory instead of alongside each dataset",
    required=False,
    type=PathPath(
        exists=True, writable=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.option(
    "--input-relative-to",
    help="Input root folder that should be used for the subfolder hierarchy in the output-base",
    required=False,
    type=PathPath(
        exists=True, writable=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.option(
    "--only-regions-in-file",
    help="Only process datasets in the given regions. Expects a file with one region code per line. "
    "(Note that some older ESA datasets have no region code, and will not match any region here.)",
    required=False,
    type=PathPath(
        exists=True, readable=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
)
@click.option(
    "--after-month",
    help="Limit the scan to datasets newer than a given month "
    "(expressed as {year}-{month}, eg '2010-01')",
    required=False,
    type=YearMonth(),
)
@click.option(
    "--before-month",
    help="Limit the scan to datasets older than the given month "
    "(expressed as {year}-{month}, eg '2010-01')",
    required=False,
    type=YearMonth(),
)
@click.option(
    "--index",
    "index_to_odc",
    is_flag=True,
    default=False,
    help="Index newly-generated metadata into the configured datacube",
)
@click.option(
    "--dry-run",
    help="Show what would be created, but don't create anything",
    is_flag=True,
    default=False,
)
@environment_option
@config_option
@pass_config(required=False)
def main(
    local_config: ODCConfig,
    output_base: Path | None,
    input_relative_to: Path | None,
    datasets: tuple[Path],
    datasets_path: Path | None,
    provider: str | None,
    overwrite_existing: bool,
    verbose: bool,
    workers: int,
    thoroughly_check_existing: bool,
    embed_location: bool | None,
    only_regions_in_file: Path | None,
    before_month: tuple[int, int] | None,
    after_month: tuple[int, int] | None,
    dry_run: bool,
    always_granule_id: bool | None,
    index_to_odc: bool,
):
    included_regions = None
    if only_regions_in_file:
        included_regions = set(only_regions_in_file.read_text().splitlines())

    if datasets_path:
        datasets = [
            *datasets,
            *(
                normalise_path(p.strip())
                for p in (datasets_path.read_text().splitlines())
            ),
        ]

    _LOG.info("kickoff", path_count=len(datasets), worker_count=workers)

    # Are we indexing on success?
    index = None
    if index_to_odc:
        _LOG.info("Indexing new datasets", local_config=local_config)
        index = index_connect(local_config, application_name="s2-prepare")
        products = {}

        def on_success(dataset: DatasetDoc, dataset_path: Path):
            """
            Index the dataset
            """
            product_name = dataset.product.name
            product = products.get(product_name)
            if not product:
                product = index.products.get_by_name(product_name)
                if not product:
                    raise ValueError(f"Product {product_name} not found in ODC index")
                products[product_name] = product

            index.datasets.add(
                Dataset(product, serialise.to_doc(dataset), uris=dataset.locations)
            )
            _LOG.debug(
                "Indexed dataset", dataset_id=dataset.id, dataset_path=dataset_path
            )

    else:

        def on_success(dataset: DatasetDoc, dataset_path: Path):
            """Nothing extra"""

    def find_inputs_in_path(input_path: Path) -> Iterable[InputDataset]:
        """
        Scan the input path for our key identifying files of a package.
        """
        found_something = False
        if provider == "sinergise.com" or not provider:
            for p in _rglob_with_self(input_path, "tileInfo.json"):
                found_something = True
                yield InputDataset(
                    producer="sinergise.com",
                    # Dataset location is the metadata file itself.
                    path=p,
                    # Output is a sibling metadata file, with the same name as the folder (usually S2A....).
                    base_folder=p.parent.parent,
                    name=p.parent.stem,
                )
        if provider == "esa.int" or not provider:
            for p in _rglob_with_self(input_path, "*.zip"):
                found_something = True
                yield InputDataset(
                    producer="esa.int",
                    # Dataset location is the zip file
                    path=p,
                    # Metadata is a sibling file with a metadata suffix.
                    base_folder=p.parent,
                    name=p.stem,
                )
        if not found_something:
            raise ValueError(
                f"No S2 datasets found in given path {input_path}. "
                f"Expected either Sinergise (productInfo.json) files or ESA zip files to be contained in it."
            )

    def find_jobs() -> Iterable[Job]:
        region_lookup = RegionLookup()

        nonlocal input_relative_to, embed_location
        for input_path in datasets:
            first = True
            for found_dataset in find_inputs_in_path(input_path):
                _LOG.debug("found_dataset", name=found_dataset.name)
                # Make sure we tick progress on extra datasets that were found.
                if not first:
                    first = False

                # Filter based on metadata
                info = found_dataset.metadata

                # Skip regions that are not in the limit?
                if included_regions or before_month or after_month:
                    if info is None:
                        raise ValueError(
                            f"Cannot filter from non-standard folder layout: {found_dataset.path} "
                            f" expected of form L1C/yyyy/yyyy-mm/area/S2_.."
                        )

                    if included_regions:
                        # If it's an older dataset without a region, try to map its area to a known region.
                        if info.region_code is None:
                            for region in region_lookup.get(info.area):
                                if region in included_regions:
                                    _LOG.debug(
                                        "mapped_area_match",
                                        input_area=info.area,
                                        region_match=region,
                                    )
                                    break
                            else:
                                _LOG.debug(
                                    "skipping.mapped_area_not_in_regions",
                                    input_area=info.area,
                                )
                                continue
                        elif info.region_code not in included_regions:
                            _LOG.debug(
                                "skipping.region_not_in_region_list",
                                region_code=info.region_code,
                            )
                            continue

                    if after_month is not None:
                        year, month = after_month

                        if info.year < year or (
                            info.year == year and info.month < month
                        ):
                            _LOG.debug(
                                "skipping.too_old",
                                dataset_year_month=(info.year, info.month),
                                max_year_month=(year, month),
                            )
                            continue
                    if before_month is not None:
                        year, month = before_month

                        if info.year > year or (
                            info.year == year and info.month > month
                        ):
                            _LOG.debug(
                                "skipping.too_young",
                                dataset_year_month=(info.year, info.month),
                                min_year_month=(year, month),
                            )
                            continue

                # Put outputs in a different folder?
                if output_base:
                    # What base folder should we choose for creating subfolders in the output?
                    if input_relative_to is None:
                        input_relative_to = _get_default_relative_folder_base(
                            found_dataset.base_folder
                        )

                    output_folder = output_base / found_dataset.base_folder.relative_to(
                        input_relative_to
                    )
                    # Default to true.
                    if embed_location is None:
                        embed_location = True
                else:
                    output_folder = found_dataset.base_folder
                    # Default to false
                    if embed_location is None:
                        embed_location = False

                # It's very slow to read the list of inner granules.
                #
                # So, if we're not thoroughly checking for missing outputs.
                if (
                    (not thoroughly_check_existing)
                    # ... and any outputs exist at all
                    and list(
                        output_folder.glob(f"{found_dataset.name}*.odc-metadata.yaml")
                    )
                    # ... and we're not overwriting our outputs
                    and not overwrite_existing
                ):
                    # Skip it!
                    _LOG.debug(
                        "At least one output exists: skipping.",
                        dataset_name=found_dataset.name,
                    )
                    continue

                # This has to read the files, so can be slow. That's why we try to skip above if possible.
                granule_offsets = found_dataset.granule_offsets

                # When granule_id is None, it means process all without filtering.
                if not granule_offsets:
                    granule_offsets = [None]
                else:
                    _LOG.debug("found_granules", granule_count=len(granule_offsets))

                for granule_offset in granule_offsets:
                    # Now that we've extracted actual granules, try again to filter by region.
                    if granule_offset and included_regions:
                        region_code = get_region_code_from_granule_offset(
                            granule_offset
                        )
                        if region_code not in included_regions:
                            _LOG.debug(
                                "skipping.granule_out_of_region",
                                granule_offset=granule_offset,
                                region_code=region_code,
                            )
                            continue

                    if always_granule_id or (
                        # None means 'auto': ie. automatically include granule id when there are multiple granules
                        always_granule_id is None and len(granule_offsets) > 1
                    ):
                        yaml_filename = (
                            f"{found_dataset.name}.{granule_offset}.odc-metadata.yaml"
                        )
                    else:
                        yaml_filename = f"{found_dataset.name}.odc-metadata.yaml"

                    output_yaml = output_folder / yaml_filename
                    if output_yaml.exists():
                        if not overwrite_existing:
                            _LOG.debug(
                                "Output exists: skipping.", output_yaml=output_yaml
                            )
                            continue

                        _LOG.debug(
                            "Output exists: overwriting.", output_yaml=output_yaml
                        )

                    _LOG.info(
                        "queued",
                        dataset_name=found_dataset.name,
                        granule=granule_offset or "any",
                    )
                    yield Job(
                        dataset_path=found_dataset.path,
                        output_yaml_path=output_yaml,
                        producer=found_dataset.producer,
                        granule_id=granule_offset,
                        embed_location=embed_location,
                    )

    errors = 0

    if dry_run:
        _LOG.info("Dry run: not writing any files.")

    # If only one process, call it directly.
    # (Multiprocessing makes debugging harder, so we prefer to make it optional)
    successes = 0

    try:
        if workers == 1 or dry_run:
            for job in find_jobs():
                try:
                    if dry_run:
                        _LOG.info(
                            "Would write dataset",
                            dataset_path=job.dataset_path,
                            output_yaml_path=job.output_yaml_path,
                        )
                    else:
                        file_already_existed = (
                            overwrite_existing and job.output_yaml_path.exists()
                        )

                        dataset, path = prepare_and_write(
                            job.dataset_path,
                            job.output_yaml_path,
                            job.producer,
                            granule_id=job.granule_id,
                            embed_location=job.embed_location,
                        )
                        _LOG.info(
                            "Wrote dataset",
                            dataset_id=dataset.id,
                            dataset_path=path,
                            output_yaml=job.output_yaml_path,
                        )

                        try:
                            on_success(dataset, path)
                        except Exception:
                            # If the post-processing function fails, we don't want this "unfinished" file still around.
                            # We'd rather let it be re-created next time.
                            if not file_already_existed:
                                _LOG.info(
                                    "Cleaning new file due to indexing failure.",
                                    output_yaml=job.output_yaml_path,
                                )
                                job.output_yaml_path.unlink()

                            raise

                    successes += 1
                except Exception:
                    _LOG.exception("failed_job", job=job)
                    errors += 1
        else:
            with Pool(processes=workers) as pool:
                for res in pool.imap_unordered(_write_dataset_safe, find_jobs()):
                    if isinstance(res, str):
                        _LOG.error(res)
                        errors += 1
                    else:
                        dataset, path = res
                        _LOG.info(
                            "Wrote dataset", dataset_id=dataset.id, dataset_path=path
                        )
                        on_success(dataset, path)
                        successes += 1
                pool.close()
                pool.join()
    finally:
        if index is not None:
            index.close()

    _LOG.info("completed", success_count=successes, failure_count=errors)
    sys.exit(errors)


def _get_default_relative_folder_base(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.lower() == "l1c":
            input_relative_to = parent
            _LOG.debug("found_base_folder", inputs_are_relative_to=input_relative_to)
            break
    else:
        raise ValueError(
            "Unknown root folder for path subfolders. "
            "(Hint: specify --input-relative-to with a parent folder of the inputs. "
            "Outputs will use the same subfolder structure.)"
        )

    return input_relative_to


def _write_dataset_safe(job: Job) -> tuple[DatasetDoc, Path] | str:
    """
    A wrapper around `prepare_and_write` that catches exceptions and makes them
    serialisable as error strings.

    (for use in multiprocessing pools etc.)
    """
    try:
        dataset, path = prepare_and_write(
            job.dataset_path,
            job.output_yaml_path,
            job.producer,
            granule_id=job.granule_id,
            embed_location=job.embed_location,
        )
        return dataset, path
    except Exception:
        return f"Failed to write dataset: {job}\n" + traceback.format_exc()


if __name__ == "__main__":
    main()
