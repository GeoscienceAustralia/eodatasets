import uuid
from pathlib import Path

import rasterio
from odc.geo.geobox import GeoBox
from shapely.geometry import box

from eodatasets3 import DatasetPrepare

_ESRI_ID_NAMESPACE = uuid.UUID("88a620e0-6b62-462b-9b23-4027e05898f3")


def as_eo3(uri: str):
    """
    Prepare an EO3 doc for the given (possibly remote ULR) ESRI TIF image.

    This is an EOD3-equivalent of ``odc-tools/apps/dc_tools/odc/apps/dc_tools/esri_land_cover_to_dc.py``

    Added mainly for testing a different use-case of the EOD3 APIs.
    """
    path = Path(uri)

    with rasterio.open(uri, GEOREF_SOURCES="INTERNAL") as opened_asset:
        geobox = GeoBox.from_rio(opened_asset)
        bbox = opened_asset.bounds
        nodata = opened_asset.nodata

    region, date_range = path.stem.split("_")
    start_date, end_date = date_range.split("-")
    if geobox.crs is None:
        print(f"Empty CRS: {uri}")
        return None

    # geom = (
    #     Geometry(box(*bbox), crs=grid_spec.crs)
    #     .to_crs("EPSG:4326", wrapdateline=True)
    #     .to_crs(grid_spec.crs)
    # )

    d = DatasetPrepare(dataset_location=uri)
    d.dataset_id = uuid.uuid5(_ESRI_ID_NAMESPACE, path.name)
    d.geometry = box(*bbox)

    d.product_name = "esri_land_cover"
    d.datetime = start_date
    d.datetime_range = start_date, end_date
    d.region_code = region.strip()
    d.processed_now()

    d.note_measurement(
        "classification",
        uri,
        relative_to_dataset_location=True,
        geobox=geobox,
        expand_valid_data=False,
        nodata=nodata,
    )

    return d.to_dataset_doc(embed_location=True, validate_correctness=False)
