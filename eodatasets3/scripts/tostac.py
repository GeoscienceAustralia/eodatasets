"""
Convert an EO3 metadata doc to a Stac Item.
"""

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from uuid import UUID

import click
from click import echo, style
from datacube.utils import jsonify_document

import eodatasets3.stac as eo3stac
from eodatasets3 import serialise
from eodatasets3.model import DatasetDoc
from eodatasets3.ui import PathPath


@click.command(help=__doc__)
@click.option("-v", "--verbose", is_flag=True)
@click.option("--stac-base-url", "-u", help="Base URL of the STAC file")
@click.option("--explorer-base-url", "-e", help="Base URL of the ODC Explorer")
@click.option(
    "--validate/--no-validate",
    default=False,
    help="Validate output STAC Item against online schemas",
)
@click.argument(
    "odc_metadata_files",
    type=PathPath(exists=True, readable=True, writable=False),
    nargs=-1,
)
def run(
    verbose: bool,
    odc_metadata_files: Iterable[Path],
    stac_base_url: str,
    explorer_base_url: str,
    validate: bool,
):
    for input_metadata in odc_metadata_files:
        dataset = serialise.from_path(input_metadata)

        name = input_metadata.stem.replace(".odc-metadata", "")
        output_path = input_metadata.with_name(f"{name}.stac-item.json")

        # Create STAC dict
        item_doc = dc_to_stac(
            dataset,
            input_metadata,
            output_path,
            stac_base_url,
            explorer_base_url,
            do_validate=False,
        )

        if validate:
            eo3stac.validate_item(item_doc)

        with output_path.open("w") as f:
            json.dump(jsonify_document(item_doc), f, indent=4, default=json_fallback)

        if verbose:
            echo(f"Wrote {style(output_path.as_posix(), 'green')}")


def dc_to_stac(
    dataset: DatasetDoc,
    input_metadata: Path,
    output_path: Path,
    stac_base_url: str,
    explorer_base_url: str,
    do_validate: bool,
) -> dict:
    """
    Backwards compatibility wrapper as some other projects started using this
    method of the script.

    It's better to call eodatasets3.stac.to_stac_item() directly.
    """

    stac_destination_url = urljoin(stac_base_url, output_path.name)

    # Following previous behaviour -- fallback to the stac destination path.
    dataset_location = (
        dataset.locations[0] if dataset.locations else stac_destination_url
    )

    doc = eo3stac.to_stac_item(
        dataset,
        stac_item_destination_url=stac_destination_url,
        # This is potentially surprising.
        #     We just assume that they're uploading the odc document to the
        #     same public folder (and with the same name.)
        #     But we need to keep it for backwards compatibility.
        odc_dataset_metadata_url=urljoin(stac_base_url, input_metadata.name),
        explorer_base_url=explorer_base_url,
        dataset_location=dataset_location,
    )
    if do_validate:
        eo3stac.validate_item(doc)

    return doc


def json_fallback(o):
    """
    This function is needed here for backwards compatibility.

    Some users try to import it from this script.
    """
    if isinstance(o, datetime):
        return f"{o.isoformat()}" if o.tzinfo else f"{o.isoformat()}Z"

    if isinstance(o, UUID):
        return str(o)

    raise TypeError(
        f"Unhandled type for json conversion: {o.__class__.__name__!r} (object {o!r})"
    )


if __name__ == "__main__":
    run()
