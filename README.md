## EO Datasets

[![Linting](https://github.com/GeoscienceAustralia/eo-datasets/actions/workflows/lint.yml/badge.svg)](https://github.com/GeoscienceAustralia/eo-datasets/actions/workflows/lint.yml)
[![Tests](https://github.com/GeoscienceAustralia/eo-datasets/actions/workflows/test.yml/badge.svg)](https://github.com/GeoscienceAustralia/eo-datasets/actions/workflows/test.yml)
[![Coverage Status](https://img.shields.io/codecov/c/github/GeoscienceAustralia/eo-datasets)](https://app.codecov.io/gh/GeoscienceAustralia/eo-datasets)

A tool to easily write, validate and convert [ODC](https://github.com/opendatacube/datacube-core)
datasets and metadata.


## Installation

    pip install eodatasets3

Python 3.10+ is supported.

## Dataset assembly

The assembler api aims to make it easy to write datasets.

```python
from datetime import datetime
from pathlib import Path

from eodatasets3 import DatasetAssembler

with DatasetAssembler(
    Path("/some/output/collection/path"), naming_conventions="default"
) as p:
    # Add some common metadata fields.
    p.platform = "landsat-7"
    p.instrument = "ETM"
    p.datetime = datetime(2019, 7, 4, 13, 7, 5)
    p.processed_now()

    # Support for custom metadata fields
    p.properties["fmask:cloud_shadow"] = 42.0

    # If you have a source dataset, you can include it as provenance.
    # Assembler can also copy common metadata properties from it.
    # (... so we didn't need to set the "platform" above!)
    p.add_source_path(source_dataset, auto_inherit_properties=True)

    # Write measurements. They can be from numpy arrays, open rasterio datasets,
    # file paths, ODC Datasets...
    p.write_measurement("red", red_path)
    ...  # now write more measurements

    # Create a jpg thumbnail image using the measurements we've written
    p.write_thumbnail(red="swir1", green="swir2", blue="red")

    # Validate the dataset and write it to the destination folder atomically.
    p.done()
```

The Assembler will write a folder of [COG](https://www.cogeo.org/) imagery, an [eo3](#open-data-cube-compatibility)
metadata doc for Open Data Cube, and create appropriate file and folder structures for the chosen naming conventions.

If you already have existing imagery, you can use DatasetAssembler to create a matching metadata document.

See [the documentation guide for more features and examples](https://eodatasets.readthedocs.io/en/latest/).

## Open Data Cube compatibility

The assembler writes a format called "eo3", which will be the native metadata format for Open Data Cube
2.0. We recommend new products are written with this format, even if targeting Open Data Cube 1.
Datacube versions from 1.8 onwards are compatible natively with eo3.

eo3 adds information about the native grid of the data, and aims to be more easily interoperable
with the upcoming [Stac Item metadata](https://github.com/radiantearth/stac-spec/tree/master/item-spec).

# Other Tools Included

## Validator


`eo3-validate` a lint-like checker to check ODC documents.

Give it ODC documents for your products, types and/or datasets to find
errors quickly.

    ❯ eo3-validate my-product.odc-product.yaml /tmp/path/to/dataset.odc-metadata.yaml
    ❯ eo3-validate https://explorer.dea.ga.gov.au/products/ga_ls_fc_3.odc-product.yaml

Note that documents are processed in order. A product should be specified before its dataset,
so that the product is known by the validator before the dataset is checked.

Similarly, a Metadata Type should ideally be specified before its product, so that the type is known.
But all are optional for basic checks

- **Note** You can use `--odc` or `--explorer-url` to automatically load a list of types and products.

### Help text

    ❯ eo3-validate --help
	Usage: eo3-validate [OPTIONS] [PATHS]...

	  Validate ODC dataset documents

	  Paths can be products, dataset documents, or directories to scan (for files
	  matching names '*.odc-metadata.yaml' etc), either local or URLs.

	  Datasets are validated against matching products that have been scanned
	  already, so specify products first, and datasets later, to ensure they can
	  be matched.

	Options:
	  --version                       Show the version and exit.
	  -W, --warnings-as-errors        Fail if any warnings are produced
	  -f, --output-format [plain|quiet|github]
					  Output format  [default: plain]
	  --thorough                      Attempt to read the data/measurements, and
					  check their properties match
	  --explorer-url TEXT             Use product definitions from the given
					  Explorer URL to validate datasets. Eg:
					  "https://explorer.dea.ga.gov.au/"
	  --odc                           Use product definitions from datacube to
					  validate datasets
	  -q, --quiet                     Only print problems, one per line
	  --help                          Show this message and exit.


### Eo3-validate github action (market place)
#### Inputs

`command`: **Required** The command to run. Default `"eo3-validate"`.
`filepath`: **Required** The filepath to odc documents. Default `""`.

#### Example usage

```
uses: actions/eo3-validate@v1
with:
  command: 'eo3-validate'
  filepath: '/'
```

### Eo3-validate github action (docker)
#### Example usage

```
name: Run eo3-validate
run: |
    docker run -v $PWD/:/code/odc-files opendatacube/eo-datasets:latest eo3-validate ./odc-files

```


### Disabling warnings

ODC is very configurable, and sometimes the validator will be too strict
for you.

You can ease some restrictions for a product by adding
a `default_allowances` section to the end of your Product definition.

(The same section can also be included in any Metadata Type, and will apply
to all products of that type)

Example supported fields:

```yaml

# Possible restriction to ease.
# All fields can be omitted (please don't include them if you don't need them).
default_allowances:

  # "It's okay if some datasets to have null geometry."
  require_geometry: false

  # Allow some metadata fields (from the metadata type) to be null
  # (but they must still be in each dataset document)
  allow_nullable_fields:
    - dataset_maturity

  # Allow some metadata fields to be entirely missing from datasets
  allow_missing_fields:
    - sentinel_product_name
    - s2cloudless_clear

  # Allow datasets to have extra measurements that weren't in the product definition.
  allow_extra_measurements: [nbar_blue, nbar_green, nbar_red]
```

### 'Thorough' mode

By default, the validator will only look at the given metadata document(s).

But when the `--thorough` flag is given, the validator will attempt to read any
referenced imagery, checking their properties match the definitions (nodata, dtype etc).

## Stac metadata conversion

`eo3-to-stac`: Convert an EO3 metadata doc to a Stac Item

	❯ eo3-to-stac --help
	Usage: eo3-to-stac [OPTIONS] [ODC_METADATA_FILES]...

	  Convert an EO3 metadata doc to a Stac Item.

	Options:
	  -v, --verbose
	  -u, --stac-base-url TEXT      Base URL of the STAC file
	  -e, --explorer-base-url TEXT  Base URL of the ODC Explorer
	  --validate / --no-validate    Validate output STAC Item against online
					schemas

	  --help                        Show this message and exit.


Example usage:

	❯ eo3-to-stac LT05_L1TP_113081_19880330_20170209_01_T1.odc-metadata.yaml
	❯ ls
	LT05_L1TP_113081_19880330_20170209_01_T1.odc-metadata.yaml
	LT05_L1TP_113081_19880330_20170209_01_T1.stac-item.json

## Prep Scripts

Some scripts are included for preparing common metadata documents,
such as landsat scenes.

`eo3-prepare`: Prepare ODC metadata from the commandline.

Some sub-commands need the ancillary dependencies, for reading from
exotic formats: `pip install .[ancillary]`

	❯ eo3-prepare --help
	Usage: eo3-prepare [OPTIONS] COMMAND [ARGS]...

	Options:
	  --version  Show the version and exit.
	  --help     Show this message and exit.

	Commands:
	  landsat-l1     Prepare eo3 metadata for USGS Landsat Level 1 data.
	  modis-mcd43a1  Prepare MODIS MCD43A1 tiles for indexing into a Data...
	  noaa-prwtr     Prepare NCEP/NCAR reanalysis 1 water pressure datasets...
	  sentinel-l1   Prepare eo3 metadata for Sentinel-2 Level 1C data produced...

Prep scripts have their own options, for example Sentinel L1 generation can filter by time
or region, if the inputs follow a common directory structure:

```
❯ eo3-prepare sentinel-l1 --help
Usage: eo3-prepare sentinel-l1 [OPTIONS] [DATASETS]...

  Prepare eo3 metadata for Sentinel-2 Level 1C data produced by Sinergise or
  ESA.

  Takes ESA zipped datasets or Sinergise dataset directories

Options:
  -v, --verbose
  -f, --datasets-path FILE        A file to read input dataset paths from, one
                                  per line
  -j, --jobs INTEGER              Number of workers to run in parallel
  --overwrite-existing / --skip-existing
                                  Overwrite if exists (otherwise skip)
  --embed-location / --no-embed-location
                                  Embed the location of the dataset in the
                                  metadata? (if you wish to store them
                                  separately. default: auto)
  --always-granule-id / --never-granule-id
                                  Include the granule id in metadata
                                  filenames? (default: auto -- include only
                                  for multi-granule files). Beware that multi-
                                  granule datasets without a granule id in the
                                  filename will overwrite each-other
  --throughly-check-existing / --cheaply-check-existing
                                  Should we open every dataset to check if
                                  *all* inner granules have been produced?
                                  Default: false.
  --provider [sinergise.com|esa.int]
                                  Restrict scanning to only packages of the
                                  given provider. (ESA assumes a zip file,
                                  sinergise a directory)
  --output-base DIRECTORY         Write metadata files into a directory
                                  instead of alongside each dataset
  --input-relative-to DIRECTORY   Input root folder that should be used for
                                  the subfolder hierarchy in the output-base
  --only-regions-in-file FILE     Only process datasets in the given regions.
                                  Expects a file with one region code per
                                  line. (Note that some older ESA datasets
                                  have no region code, and will not match any
                                  region here.)
  --after-month YEAR-MONTH        Limit the scan to datasets newer than a
                                  given month (expressed as {year}-{month}, eg
                                  '2010-01')
  --before-month YEAR-MONTH       Limit the scan to datasets older than the
                                  given month (expressed as {year}-{month}, eg
                                  '2010-01')
  --index                         Index newly-generated metadata into the
                                  configured datacube
  --dry-run                       Show what would be created, but don't create
                                  anything
  -E, --env TEXT
  -C, --config, --config_file TEXT
  --help                          Show this message and exit.
```

An example of preparing metadata in a separate directory (not alongside the datasets) at NCI
can be as follows:

```bash
module use -a /g/data/v10/private/modules/modulefiles /g/data/v10/public/modules/modulefiles
module load eodatasets3

# With a folder of input paths, 4 workers, and separate output directory:
eo3-prepare sentinel-l1 -j 4 --output-base /output/metadata/directory \
  /g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2021

# Using a file for input paths. Filter them to a certain region list and recent months:
eo3-prepare sentinel-l1 \
    --output-base /g/data/v10/agdc/jez/c3/L1C  \
    --only-regions-in-file test-regions.txt \
    --after-month 2022-04 \
    -f l1cs-2022-05-02.txt

```


`eo3-package-wagl`: Convert and package WAGL HDF5 outputs.

 Needs the wagl dependencies group: `pip install .[wagl]`

	❯ eo3-package-wagl --help
	Usage: eo3-package-wagl [OPTIONS] H5_FILE

	  Package WAGL HDF5 Outputs

	  This will convert the HDF5 file (and sibling fmask/gqa files) into
	  GeoTIFFS (COGs) with datacube metadata using the DEA naming conventions
	  for files.

	Options:
	  --level1 FILE                   Optional path to the input level1 metadata
					  doc (otherwise it will be loaded from the
					  level1 path in the HDF5)

	  --output DIRECTORY              Put the output package into this directory
					  [required]

	  -p, --product [nbar|nbart|lambertian|sbt]
					  Package only the given products (can specify
					  multiple times)

	  --with-oa / --no-oa             Include observation attributes (default:
					  true)

	  --with-oa / --no-oa             Include observation attributes (default:
					  true)

	  --oa-resolution FLOAT           Resolution choice for observation attributes
					  (default: automatic based on sensor)

	  --help                          Show this message and exit.


# Development Setup

Run the tests using [pytest](http://pytest.org/).

	❯ pytest

You may need to install test dependencies first:

	❯ pip install -e .[test]

Dependencies such as gdal can be tricky to install on some systems. You
may prefer to use the included Docker file for development: run `make
build` to create a container, and `make test` to run tests.

We have strict linting and formatting checks on this reposistory, so
please run pre-commit (below) after checkout.

## Pre-commit setup

	❯ pip install pre-commit
	❯ pre-commit install

(if you are using Conda, you need to `conda install pre_commit` instead of using pip)

Your code will now be formatted and validated before each commit. You can also invoke it manually by running `pre-commit run`

This allows you to immediately catch and fix issues before you raise a pull request that fails.

Most notably, all code is formatted using
[black](https://github.com/ambv/black), and checked with
[pyflakes](https://github.com/PyCQA/pyflakes).


## Docker dependencies

To update the set of frozen dependencies inside docker:

1) Ensure you have an existing, working image. ie, run `make build`. If the current image is broken,
you may need to `git checkout` the last working version first.
2) Run `make dependency-update`. This will recalculate the list of all dependencies from the definitions in setup.py.

Note that this will run pip-compile _inside_ the docker container for maximum compatibility, hence the need for an existing container.

## Creating Releases

First, draft [some release notes](https://github.com/GeoscienceAustralia/eo-datasets/releases)
for users of the library.

Now tag and upload:

```
# Be up-to-date.
git fetch origin

# Create a tag for the new version
# (using semantic versioning https://semver.org/)
git tag eodatasets3-<version> origin/eodatasets3

# Create package
python3 setup.py sdist bdist_wheel

# Upload it (Jeremy, Damien, Kirill have pypi ownership)
python3 -m twine upload  dist/*

# Push tag to main repository
git push origin --tags
```
