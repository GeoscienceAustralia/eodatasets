#!/usr/bin/env python

from posixpath import join as ppjoin
from pathlib import Path
import numpy
import h5py
import rasterio
from rasterio.crs import CRS
from affine import Affine
from eodatasets3 import DatasetAssembler, images, utils
import eodatasets3.wagl
from eodatasets3.serialise import loads_yaml
from boltons.iterutils import get_path


GDAL_H5_FMT = 'HDF5:"{filename}":/{dataset_pathname}'


def package_non_standard(outdir, granule):
    """
    yaml creator for the ard pipeline.
    Purely for demonstration purposes only.
    Can easily be expanded to include other datasets.

    A lot of the metadata such as date, can be extracted from the yaml
    doc contained within the HDF5 file at the following path:

    [/<granule_id>/METADATA/CURRENT]
    """

    outdir = Path(outdir)
    out_fname = Path(str(granule.wagl_hdf5).replace("wagl.h5", "yaml"))

    with DatasetAssembler(metadata_path=out_fname, naming_conventions="dea") as da:
        level1 = granule.source_level1_metadata
        da.add_source_dataset(level1, auto_inherit_properties=True)
        da.product_family = "ard"
        da.producer = "ga.gov.au"

        with h5py.File(granule.wagl_hdf5, "r") as fid:
            img_paths = [
                ppjoin(fid.name, pth)
                for pth in eodatasets3.wagl._find_h5_paths(fid, "IMAGE")
            ]
            granule_group = fid[granule.name]

            try:
                wagl_path, *ancil_paths = [
                    pth
                    for pth in (
                        eodatasets3.wagl._find_h5_paths(granule_group, "SCALAR")
                    )
                    if "METADATA" in pth
                ]
            except ValueError:
                raise ValueError("No nbar metadata found in granule")

            [wagl_doc] = loads_yaml(granule_group[wagl_path][()])

            da.processed = get_path(wagl_doc, ("system_information", "time_processed"))

            org_collection_number = utils.get_collection_number(
                da.producer, da.properties["landsat:collection_number"]
            )

            da.dataset_version = f"{org_collection_number}.1.0"
            da.region_code = eodatasets3.wagl._extract_reference_code(da, granule.name)

            eodatasets3.wagl._read_gqa_doc(da, granule.gqa_doc)
            eodatasets3.wagl._read_fmask_doc(da, granule.fmask_doc)

            if granule.fmask_image:
                da.note_measurement(
                    "oa_fmask", granule.fmask_image, expand_valid_data=False,
                )

            for pathname in img_paths:
                ds = fid[pathname]
                ds_path = Path(ds.name)

                # eodatasets internally uses this grid spec to group image datasets
                grid_spec = images.GridSpec(
                    shape=ds.shape,
                    transform=Affine.from_gdal(*ds.attrs["geotransform"]),
                    crs=CRS.from_wkt(ds.attrs["crs_wkt"]),
                )

                pathname = GDAL_H5_FMT.format(
                    filename=str(outdir.joinpath(granule.wagl_hdf5)),
                    dataset_pathname=pathname,
                )

                # product group name; lambertian, nbar, nbart, oa
                if "STANDARDISED-PRODUCTS" in str(ds_path):
                    product_group = ds_path.parent.name
                else:
                    product_group = "oa"

                # spatial resolution group
                # used to separate measurements with the same name
                resolution_group = "rg{}".format(ds_path.parts[2].split("-")[-1])

                measurement_name = (
                    "_".join(
                        [
                            resolution_group,
                            product_group,
                            ds.attrs.get("alias", ds_path.name),
                        ]
                    )
                    .replace("-", "_")
                    .lower()
                )  # we don't wan't hyphens in odc land

                # include this band in defining the valid data bounds?
                include = True if "nbart" in measurement_name else False

                # this method will not give as the transform and crs and eodatasets will complain later
                # TODO: raise an issue on github for eodatasets
                # da.note_measurement(
                #     measurement_name,
                #     pathname,
                #     expand_valid_data=False,
                # )

                no_data = ds.attrs.get("no_data_value")
                if no_data is None:
                    no_data = float("nan")

                # if we are of type bool, we'll have to convert just for GDAL
                if ds.dtype.name == "bool":
                    no_data = 255
                    img_out_fname = outdir.joinpath("{}.tif".format(measurement_name))
                    kwargs = {
                        "driver": "GTiff",
                        "dtype": "uint8",
                        "count": 1,
                        "height": ds.shape[0],
                        "width": ds.shape[1],
                        "crs": CRS.from_wkt(ds.attrs["crs_wkt"]),
                        "transform": Affine.from_gdal(*ds.attrs["geotransform"]),
                        "nodata": no_data,
                        "compress": "deflate",
                        "zlevel": 4,
                        "predictor": 2,
                        "blockxsize": ds.chunks[1],
                        "blockysize": ds.chunks[0],
                        "tiled": "yes",
                    }
                    with rasterio.open(img_out_fname, "w", **kwargs) as out_ds:
                        out_ds.write(numpy.uint8(ds[:], 1))

                    da.note_measurement(
                        measurement_name, img_out_fname, expand_valid_data=include
                    )
                else:
                    # work around as note_measurement doesn't allow us to specify the gridspec
                    da._measurements.record_image(
                        measurement_name,
                        grid_spec,
                        pathname,
                        ds[:],
                        nodata=no_data,
                        expand_valid_data=include,
                    )

        # the longest part here is generating the valid data bounds vector
        # landsat 7 post SLC-OFF can take a really long time
        return da.done()
