# coding=utf-8
from __future__ import absolute_import
import functools
import os
import shutil
import logging
import time
from subprocess import check_call
import datetime
import uuid
import socket

from pathlib import Path
from eodatasets import serialise, verify, drivers, metadata
from eodatasets.browseimage import create_dataset_browse_images
import eodatasets.type as ptype


GA_CHECKSUMS_FILE_NAME = 'package.sha1'

_LOG = logging.getLogger(__name__)

_RUNTIME_ID = uuid.uuid1()


def init_locally_processed_dataset(software_provenance, uuid=None):
    """
    Create a blank dataset for a newly created dataset on this machine.
    :type software_provenance: eodatasets.provenance.SoftwareProvenance
    :param uuid: The existing dataset_id, if any.
    :rtype: ptype.DatasetMetadata
    """
    md = ptype.DatasetMetadata(
        id_=uuid,
        lineage=ptype.LineageMetadata(
            machine=ptype.MachineMetadata(
                hostname=socket.getfqdn(),
                runtime_id=_RUNTIME_ID,
                software=software_provenance,
                uname=' '.join(os.uname())
            ),
        )
    )
    return md


def init_existing_dataset(software_provenance=None, uuid=None, source_hostname=None):
    """
    Init a dataset of mostly unknown origin.

    Source hostname can be supplied if known.

    :param uuid: The existing dataset_id, if any.
    :param source_hostname: Hostname where processed, if known.
    :rtype: ptype.DatasetMetadata
    """
    md = ptype.DatasetMetadata(
        id_=uuid,
        lineage=ptype.LineageMetadata(
            machine=ptype.MachineMetadata(
                hostname=source_hostname,
                software=software_provenance
            )
        )
    )
    return md


def _copy_file(source_path, destination_path, compress_imagery=True, hard_link=False):
    """
    Copy a file from source to destination if needed. Maybe apply compression.

    (it's generally faster to compress during a copy operation than as a separate step)

    :type source_path: Path
    :type destination_path: Path
    :type compress_imagery: bool
    :type hard_link: bool
    :return: Size in bytes of destination file.
    :rtype int
    """

    source_file = str(source_path)
    destination_file = str(destination_path)

    # Copy to destination path.
    original_suffix = source_path.suffix.lower()
    suffix = destination_path.suffix.lower()

    if destination_path.exists():
        _LOG.info('Destination exists: %r', destination_file)
    elif (original_suffix == suffix) and hard_link:
        _LOG.info('Hard linking %r -> %r', source_file, destination_file)
        os.link(source_file, destination_file)
    # If a tif image, compress it losslessly.
    elif suffix == '.tif' and compress_imagery:
        _LOG.info('Copying compressed %r -> %r', source_file, destination_file)
        check_call(
            [
                'gdal_translate',
                '--config', 'GDAL_CACHEMAX', '512',
                '--config', 'TILED', 'YES',
                '-co', 'COMPRESS=lzw',
                source_file, destination_file
            ]
        )
    else:
        _LOG.info('Copying %r -> %r', source_file, destination_file)
        shutil.copy(source_file, destination_file)

    return destination_path.stat().st_size


def prepare_target_imagery(
        image_directory,
        package_directory,
        to_band_fn,
        compress_imagery=True,
        hard_link=False,
        after_file_copy=lambda file_path: None):
    """
    Copy a directory of files if not already there. Possibly compress images.

    :type to_band_fn: (str) -> ptype.BandMetadata
    :type image_directory: Path
    :type package_directory: Path
    :type after_file_copy: Path -> None
    :type hard_link: bool
    :type compress_imagery: bool
    :return: Total size of imagery in bytes, band dictionary.
    :rtype (int, dict[str, ptype.BandMetadata])
    """
    if not package_directory.exists():
        package_directory.mkdir()

    size_bytes = 0
    bands = {}
    for source_path in image_directory.iterdir():
        # Skip hidden files
        if source_path.name.startswith('.'):
            continue

        band = to_band_fn(source_path)
        if band is None:
            continue

        band.path = ptype.rebase_path(image_directory, package_directory, band.path)

        size_bytes += _copy_file(source_path, band.path, compress_imagery, hard_link=hard_link)

        bands[band.number] = band
        after_file_copy(band.path)

    return size_bytes, bands


class IncompletePackage(Exception):
    """
    Package is incomplete: (eg. Not enough metadata could be found.)
    """
    pass


def package_existing_dataset(dataset_driver,
                             image_directory,
                             target_directory,
                             hard_link=False,
                             source_datasets=None):
    """
    Package an existing dataset folder (with mostly unknown provenance).

    This is intended for old datasets where little information was recorded.

    For brand new datasets, it's better to use package_dataset() with a dataset created via
     init_locally_processed_dataset() to capture local machine information.

    :param hard_link:
        Hard link imagery files instead of copying them. Much faster.

    :type dataset_driver: drivers.DatasetDriver
    :type image_directory: Path or str
    :type target_directory: Path or str
    :type hard_link: bool
    :type source_datasets: dict of (str, ptype.DatasetMetadata)

    :raises IncompletePackage:
        Mot enough metadata can be extracted from the dataset.

    :return: The generated GA Dataset ID (ga_label)
    :rtype: str
    """
    d = init_existing_dataset()
    d.lineage.source_datasets = source_datasets

    #: :type: ptype.DatasetMetadata
    d = dataset_driver.fill_metadata(d, image_directory)

    return package_dataset(dataset_driver, d, image_directory, target_directory, hard_link=hard_link)


def package_dataset(dataset_driver,
                    dataset,
                    image_directory,
                    target_directory,
                    hard_link=False):
    """
    Package the given dataset folder.
    :type hard_link: bool
    :type dataset_driver: drivers.DatasetDriver
    :type dataset: ptype.Dataset
    :type image_directory: Path or str
    :type target_directory: Path or str

    :raises IncompletePackage: If not enough metadata can be extracted from the dataset.
    :return: The generated GA Dataset ID (ga_label)
    :rtype: str
    """
    start = time.time()
    checksums = verify.PackageChecksum()

    target_path = Path(target_directory).absolute()
    image_path = Path(image_directory).absolute()

    target_metadata_path = serialise.expected_metadata_path(target_path)
    if target_metadata_path.exists():
        _LOG.info('Already packaged? Skipping %s', target_path)
        return
    target_checksums_path = target_path / GA_CHECKSUMS_FILE_NAME

    _LOG.debug('Packaging %r -> %r', image_path, target_path)
    if image_path.resolve() != target_path.resolve():
        package_directory = target_path.joinpath('package')
    else:
        package_directory = target_path

    size_bytes, bands = prepare_target_imagery(
        image_path,
        package_directory,
        to_band_fn=functools.partial(dataset_driver.to_band, dataset),
        after_file_copy=checksums.add_file,
        hard_link=hard_link
    )
    dataset.image.bands.update(bands)

    #: :type: ptype.DatasetMetadata
    dataset = ptype.rebase_paths(image_path, package_directory, dataset)

    if not target_path.exists():
        target_path.mkdir()

    # TODO: Add proper validation to dataset structure.
    if not dataset.platform or not dataset.platform.code:
        raise IncompletePackage('Incomplete dataset. Not enough metadata found: %r' % dataset)

    dataset.product_type = dataset_driver.get_id()
    dataset.ga_label = dataset_driver.get_ga_label(dataset)
    dataset.size_bytes = size_bytes
    dataset.checksum_path = target_checksums_path
    if not dataset.creation_dt:
        # Default creation time is creation of the source folder.
        dataset.creation_dt = datetime.datetime.utcfromtimestamp(image_path.stat().st_ctime)

    d = metadata.expand_common_metadata(dataset)

    create_dataset_browse_images(
        dataset_driver,
        d,
        target_path,
        after_file_creation=checksums.add_file
    )

    target_metadata_path = serialise.write_dataset_metadata(target_path, d)
    checksums.add_file(target_metadata_path)

    checksums.write(target_checksums_path)
    _LOG.info('Packaged in %.02f: %s', time.time() - start, target_metadata_path)

    return d.ga_label
