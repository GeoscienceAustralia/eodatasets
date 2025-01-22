import re
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Dict, Mapping, Optional, Sequence, Set, Union
from urllib.parse import quote, unquote, urlparse

import datacube.utils.uris as dc_uris

from eodatasets3 import utils
from eodatasets3.model import DEA_URI_PREFIX, Location
from eodatasets3.properties import Eo3Dict, Eo3Interface

# Needed when packaging zip or tar files.
dc_uris.register_scheme("zip", "tar")


class LazyProductName:
    def __init__(self, include_instrument=True, include_collection=False) -> None:
        super().__init__()
        self.include_instrument = include_instrument
        self.include_collection = include_collection

    def __get__(self, c: "NamingConventions", owner) -> str:
        if c.metadata.product_name:
            return c.metadata.product_name

        instrument = c.instrument_abbreviated if self.include_instrument else ""
        return "_".join(
            p
            for p in (
                c.producer_abbreviated,
                f"{c.platform_abbreviated or ''}{instrument or ''}",
                c.metadata.product_family,
                (
                    c.metadata.product_maturity
                    if c.metadata.product_maturity != "stable"
                    else None
                ),
                (
                    f"{c.displayed_collection_number}"
                    if (
                        self.include_collection
                        and c.displayed_collection_number is not None
                    )
                    else None
                ),
            )
            if p
        )


def _strip_major_version(version: str) -> str:
    """
    >>> _strip_major_version('1.2.3')
    '2.3'
    >>> _strip_major_version('01.02.03')
    '02.03'
    >>> _strip_major_version('30.40')
    '40'
    >>> _strip_major_version('40')
    ''
    """
    return ".".join(version.split(".")[1:])


class LazyLabel:
    def __init__(self, include_version=True, strip_major_version=False) -> None:
        super().__init__()
        self.strip_major_version = strip_major_version
        self.include_version = include_version

    def __get__(self, c: "NamingConventions", owner) -> str:
        d = c.metadata

        product_prefix = c.product_name

        version = d.dataset_version
        if version and self.include_version:
            if self.strip_major_version:
                version = _strip_major_version(version)
            version = version.replace(".", "-")
            product_prefix = f"{c.product_name}-{version}"

        maturity: str = d.properties.get("dea:dataset_maturity")
        return "_".join(
            [
                p
                for p in (
                    product_prefix,
                    d.region_code,
                    f"{d.datetime:%Y-%m-%d}",
                    maturity,
                )
                if p
            ]
        )


class LazyPlatformAbbreviation:
    # The abbreviations mentioned in DEA naming conventions doc.
    KNOWN_PLATFORM_ABBREVIATIONS: ClassVar[Dict[str, str]] = {
        "landsat-5": "ls5",
        "landsat-7": "ls7",
        "landsat-8": "ls8",
        "landsat-9": "ls9",
        "sentinel-1a": "s1a",
        "sentinel-1b": "s1b",
        "sentinel-2a": "s2a",
        "sentinel-2b": "s2b",
        "sentinel-2c": "s2c",
        "aqua": "aqu",
        "terra": "ter",
    }

    # If all platform (abbreviations) match a pattern, return this group name instead.
    KNOWN_PLATFORM_GROUPINGS: ClassVar[Dict[str, re.Pattern]] = {
        "ls": re.compile(r"ls\d+"),
        "s1": re.compile(r"s1[a-z]+"),
        "s2": re.compile(r"s2[a-z]+"),
    }

    def __init__(
        self,
        *,
        known_abbreviations: Dict = None,
        grouped_abbreviations: Dict = None,
        show_specific_platform=True,
        allow_unknown_abbreviations=True,
    ) -> None:
        self.known_abbreviations = (
            known_abbreviations or self.KNOWN_PLATFORM_ABBREVIATIONS
        )
        self.grouped_abbreviations = (
            grouped_abbreviations or self.KNOWN_PLATFORM_GROUPINGS
        )
        self.show_specific_platform = show_specific_platform

        self.allow_unknown_abbreviations = allow_unknown_abbreviations

    def __get__(self, c: "NamingConventions", owner) -> Optional[str]:
        """Abbreviated form of a satellite, as used in dea product names. eg. 'ls7'."""

        p = c.metadata.platforms
        if not p:
            return None

        if not self.allow_unknown_abbreviations:
            unknowns = p.difference(self.known_abbreviations)
            if unknowns:
                raise ValueError(
                    f"We don't know the DEA abbreviation for platforms {unknowns!r}. "
                    f"We'd love to add more! Raise an issue on Github: "
                    f"https://github.com/GeoscienceAustralia/eo-datasets/issues/new' "
                )

        abbreviations = sorted(
            self.known_abbreviations.get(s, s.replace("-", "")) for s in p
        )

        if self.show_specific_platform and len(abbreviations) == 1:
            return abbreviations[0]

        # If all abbreviations are in a group, name it using that group.
        # (eg. "ls" instead of "ls5-ls7-ls8")
        for group_name, pattern in self.grouped_abbreviations.items():
            if all(pattern.match(a) for a in abbreviations):
                return group_name

        # Otherwise, there's a mix of platforms.

        # Is there a common constellation?
        constellation = c.metadata.properties.get("constellation")
        if constellation:
            return constellation

        # Don't bother to include platform in name for un-groupable mixes of them.
        if not self.allow_unknown_abbreviations:
            raise NotImplementedError(
                f"Satellite constellation abbreviation is not known for platforms {p}. "
                f"(for DEA derivative naming conventions.)"
                f"    Is this a mistake? We'd love to add more! Raise an issue on Github: "
                f"https://github.com/GeoscienceAustralia/eo-datasets/issues/new' "
            )
        return None


class LazyInstrumentAbbreviation:
    def __get__(self, c: "NamingConventions", owner) -> Optional[str]:
        """Abbreviated form of an instrument name, as used in dea product names. eg. 'c'."""
        if not c.metadata.instrument:
            return None

        platforms = c.metadata.platforms
        if not platforms or len(platforms) > 1:
            return None

        [p] = platforms

        if p.startswith("sentinel-1") or p.startswith("sentinel-2"):
            return c.metadata.instrument[0].lower()

        if p.startswith("landsat"):
            # Extract from usgs standard:
            # landsat:landsat_product_id: LC08_L1TP_091075_20161213_20170316_01_T2
            # landsat:landsat_scene_id: LC80910752016348LGN01
            landsat_id = c.metadata.properties.get("landsat:landsat_product_id")
            if landsat_id is None:
                landsat_id = c.metadata.properties.get("landsat:landsat_scene_id")

            # from USGS STAC, label is LC08_L2SP_178079_20210417_20210424_02_T1_SR and
            # landsat:scene_id: LC81780792021107LGN00
            if landsat_id is None:
                landsat_id = c.metadata.properties.get("landsat:scene_id")

            if not landsat_id:
                raise NotImplementedError(
                    "No landsat scene or product id: cannot abbreviate Landsat instrument."
                )

            return landsat_id[1].lower()

        # Otherwise, it's unknown.
        raise NotImplementedError(
            f"Instrument abbreviations aren't supported for platform {p!r}. "
            f"We'd love to add more support! Raise an issue on Github: "
            f"https://github.com/GeoscienceAustralia/eo-datasets/issues/new' "
        )


class LazyProducerAbbreviation:
    KNOWN_PRODUCER_ABBREVIATIONS: ClassVar[Dict[str, str]] = {
        "ga.gov.au": "ga",
        "usgs.gov": "usgs",
        "sinergise.com": "sinergise",
        "digitalearthafrica.org": "deafrica",
        "esa.int": "esa",
        # Is there another organisation you want to use? Pull requests very welcome!
    }

    def __init__(self, *, known_abbreviations: Dict = None) -> None:
        self.known_abbreviations = (
            known_abbreviations or self.KNOWN_PRODUCER_ABBREVIATIONS
        )

    def __get__(self, c: "NamingConventions", owner) -> Optional[str]:
        """Abbreviated form of a producer, as used in dea product names. eg. 'ga', 'usgs'."""
        if not c.metadata.producer:
            return None

        try:
            return self.known_abbreviations[c.metadata.producer]
        except KeyError:
            raise NotImplementedError(
                f"We don't know how to abbreviate organisation domain name {c.metadata.producer!r}. "
                f"We'd love to add more orgs! Raise an issue on Github: "
                f"https://github.com/GeoscienceAustralia/eo-datasets/issues/new' "
            )


class LazyRegionOffset:
    def __get__(self, c: "NamingConventions", owner) -> Optional[str]:
        # Cut the region code in subfolders
        region_code = c.metadata.region_code
        if region_code:
            return "/".join(utils.subfolderise(region_code))
        return None


class LazyTimeOffset:
    def __init__(self, date_folders_format="%Y/%m/%d") -> None:
        self.date_folders_format = date_folders_format

    def __get__(self, c: "NamingConventions", owner) -> Optional[str]:
        return c.metadata.datetime.strftime(self.date_folders_format)


class LazyDestinationFolder:
    def __init__(
        self,
        include_version=False,
        include_non_final_maturity=True,
    ) -> None:
        super().__init__()
        self.include_version = include_version
        self.include_non_final_maturity = include_non_final_maturity

    def __get__(self, c: "NamingConventions", owner) -> str:
        """The folder hierarchy the datasets files go into.

        This is returned as a relative path.

        (forward slashes, but no starting or ending slash)

        Example: ``"ga_ls8c_ard_3/092/084/2016/06/28"``
        """
        d = c.metadata
        parts = [c.product_name]

        if self.include_version:
            parts.append(d.dataset_version.replace(".", "-"))

        if c.region_folder is not None:
            parts.append(c.region_folder)
        if c.time_folder is not None:
            parts.append(c.time_folder)

        if self.include_non_final_maturity:
            # If it's not a final product, append the maturity to the folder.
            maturity: str = d.properties.get("dea:dataset_maturity")
            if maturity and maturity != "final":
                parts[-1] = f"{parts[-1]}_{maturity}"

        if c.dataset_separator_field is not None:
            val = d.properties[c.dataset_separator_field]
            # TODO: choosable formatter?
            if isinstance(val, datetime):
                val = f"{val:%Y%m%dT%H%M%S}"
            parts.append(val)
        return Path(*parts).as_posix()


class LazyDatasetLocation:
    """The location of the dataset as indexed into ODC. Defaults to the metadata path."""

    def __get__(self, c: "NamingConventions", owner) -> str:
        if not c.collection_prefix:
            raise ValueError(
                "collection_prefix is required if you're not setting a "
                "dataset_location or metadata_path!"
            )

        offset = c.dataset_folder
        if Path(offset).is_absolute():
            raise ValueError("Dataset offset is expected to be relative to collection")
        return f"{c.collection_prefix}/{offset}/"


class MissingRequiredFieldsError(ValueError): ...


class RequiredPropertyDict(Eo3Dict):
    """A wrapper for Eo3 Dict that throws a loud error if a required field is accessed
     by not yet set.

    - It gives a friendly error of what fields are required, rather than the user
    seeing obtuse "None" errors sprinkled throughout their code.
    - It will _only_ complain about the given required fields. Other fields behave
      like a normal dictionary.

    """

    # Displayed to user for friendlier errors.
    _REQUIRED_PROPERTY_HINTS: ClassVar[Dict[str, str]] = {
        "odc:product_family": 'eg. "wofs" or "level1"',
        "odc:processing_datetime": "Time of processing, perhaps datetime.utcnow()?",
        "odc:producer": "Creator of data, eg 'usgs.gov' or 'ga.gov.au'",
        "odc:dataset_version": "eg. 1.0.0",
    }

    def __init__(
        self,
        required_fields: Set[str],
        properties=None,
    ) -> None:
        self.required_fields = required_fields
        super().__init__(properties)

    def __getitem__(self, item):
        try:
            val = super().__getitem__(item)
            if (not val) and (item in self.required_fields):
                self._raise_all_missing_requirements()
            return val
        except KeyError:
            if item in self.required_fields:
                self._raise_all_missing_requirements()
            raise

    def _raise_all_missing_requirements(self):
        """
        Do we have enough properties to generate file or product names?
        """
        missing_props = []
        for f in self.required_fields:
            if f not in self._props:
                missing_props.append(f)
        if missing_props:
            examples = []
            for p in sorted(missing_props):
                hint = self._REQUIRED_PROPERTY_HINTS.get(p, "")
                if hint:
                    hint = f" ({hint})"
                examples.append(f"\n- {p!r}{hint}")

            raise MissingRequiredFieldsError(
                f"Need more properties to fulfill naming conventions."
                f"{''.join(examples)}"
            )


class EnforceRequirementProperties(Eo3Interface):
    """
    A wrapper for EO3 fields that throws a loud error if a field in required_properties isn't set.

    Throws MissingRequiredFields error.
    """

    @property
    def properties(self) -> Eo3Dict:
        return self._props

    def __init__(self, properties: Mapping, required_fields: Set[str]) -> None:
        self._props = RequiredPropertyDict(required_fields, properties)


class LazyFileName:
    def __init__(self, file_id: str, suffix: str) -> None:
        self.file_id = file_id
        self.suffix = suffix

    def __get__(self, c: "NamingConventions", owner) -> str:
        return c.filename(file_id=self.file_id, suffix=self.suffix)


class LazyProductURI:
    def __get__(self, n: "NamingConventions", owner) -> Optional[str]:
        if not n.base_product_uri:
            return None

        return f"{n.base_product_uri}/product/{quote(n.product_name)}"


def resolve_location(path: Location) -> str:
    """
    Make sure a dataset location is a URL, suitable to be
    the dataset_location in datacube indexing.

    Users may specify a pathlib.Path(), and we'll convert it as needed.
    """
    if isinstance(path, str):
        if not dc_uris.is_url(path) and not dc_uris.is_vsipath(path):
            raise ValueError(
                "A string location is expected to be a URL or VSI path. "
                "Perhaps you want to give it as a local pathlib.Path()?"
            )
        return path

    path = dc_uris.normalise_path(path)
    if ".tar" in path.suffixes:
        return f"tar:{path}!/"
    elif ".zip" in path.suffixes:
        return f"zip:{path}!/"
    else:
        uri = unquote(path.as_uri())
        # Base paths specified as directories must end in a slash,
        # so they will be url joined as subfolders. (pathlib strips them)
        if path.is_dir():
            return f"{uri}/"
        return uri


def _as_path(url: str) -> Path:
    """Try to convert the given URL to a local Path"""
    parts = urlparse(url)
    if not parts.scheme == "file":
        raise ValueError(f"Expected a filesystem path, got a URL! {url!r}")

    return Path(parts.path)


class NamingConventions:
    """
    A generator of names for products, data labels, file paths, urls, etc.

    These are generated based on a given set of naming conventions, but a user
    can manually override any properties to avoid generation.

    Create an instance by calling :meth:`eodatasets3.namer`:

    .. testcode ::

        from eodatasets3 import namer

        properties = {
            'eo:platform': 'sentinel-2a',
            'eo:instrument': 'MSI',
            'odc:product_family': 'level1',
        }
        n = namer(properties, conventions='default')
        print(n.product_name)

    .. testoutput ::

       s2am_level1

    .. note ::

       You may want to use an :class:`eodatasets3.DatasetDoc` instance rather than a dict for
       properties, to get convenience methods such as ``.platform = 'sentinel-2a'``, `.properties`, automatic
       property normalisation etc.

       See :ref:`the naming section<names_n_paths>` for an example.

    Fields are lazily generated when accessed using the underlying metadata properties,
    but you can manually set any field to avoid generation:

    .. doctest ::

        >>> from eodatasets3 import DatasetDoc
        >>> p = DatasetDoc()
        >>> p.platform = 'landsat-7'
        >>> p.product_family = 'nbar'
        >>>
        >>> n = namer(conventions='default', properties=p)
        >>> n.product_name
        'ls7_nbar'
        >>> # Manually override the abbreviation:
        >>> n.platform_abbreviated = 'ls'
        >>> n.product_name
        'ls_nbar'
        >>> # Or manually set the entire product name to avoid generation:
        >>> n.product_name = 'custom_nbar_albers'

    In order to calculate paths, give it a collection prefix. This can be a :class:`Path object <pathlib.Path>`
    for local files, or a URL str for remote.

    .. doctest ::

       >>> p.datetime = datetime(2014, 4, 5)
       >>> collection = "s3://dea-public-data-dev/collections"
       >>> n = namer(conventions='default', properties=p, collection_prefix=collection)
       >>> n.dataset_location
       's3://dea-public-data-dev/collections/ls7_nbar/2014/04/05/'
       >>> n.metadata_file
       'ls7_nbar_2014-04-05.odc-metadata.yaml'

    All fields named ``*_file`` are filenames inside (relative to) the
    ``self.dataset_location``.

        >>> n.resolve_file('thumbnail.jpg')
        's3://dea-public-data-dev/collections/ls7_nbar/2014/04/05/thumbnail.jpg'

    """

    _ABSOLUTE_MINIMAL_PROPERTIES: ClassVar[Set[str]] = {
        "odc:product_family",
        # Required by Stac regardless.
        "datetime",
    }

    # Placed here for public usage, as people can extend the defaults.
    KNOWN_PRODUCER_ABBREVIATIONS = LazyProducerAbbreviation.KNOWN_PRODUCER_ABBREVIATIONS
    KNOWN_PLATFORM_ABBREVIATIONS = LazyPlatformAbbreviation.KNOWN_PLATFORM_ABBREVIATIONS
    KNOWN_PLATFORM_GROUPINGS = LazyPlatformAbbreviation.KNOWN_PLATFORM_GROUPINGS

    # These are lazily computed on read if not overridden by the user.
    # ie. User can set th names.product_name = 'blah'

    #: Product name for ODC
    #:
    product_name: str = LazyProductName(include_collection=True)
    #: Identifier URL for the product
    #: (This is seen as a global id for the product, unlike the plain product
    #: name. It doesn't have to resolve to a real path)
    #:
    #: Eg. ``https://collections.dea.ga.gov.au/product/ga_ls8c_ard_3``
    #:
    product_uri: str = LazyProductURI()

    #: Abbreviated form of the platform, used in most other
    #: paths and names here.
    #:
    #: For example, ``landsat-7`` is usually abbreviated to ``ls7``
    platform_abbreviated: str = LazyPlatformAbbreviation()

    #: Abbreviated form of the instrument, used in most other
    #: paths and names here.
    #:
    #: For example, ``ETM+`` is usually abbreviated to ``e``
    instrument_abbreviated: str = LazyInstrumentAbbreviation()
    #: Abbreviated form of the producer of the dataset
    #: (the producing organisation)
    #:
    #: For example, "ga.gov.au" is abbreviated to "ga"
    producer_abbreviated: str = LazyProducerAbbreviation()

    #: The Label for the dataset. This is a human-readable alternative
    #: to showing the UUID in most parts of ODC. It's used by default in filenames
    #:
    # No major version by default, as the product name contains it (the collection version).
    dataset_label: str = LazyLabel(strip_major_version=True)

    #: The pattern for generating file names.
    #:
    #: The pattern is in python's ``str.format()`` syntax,
    #: with fields ``{file_id}`` and ``{suffix}``
    #:
    #: The namer instance is readable from ``{n}``.
    filename_pattern: str = "{n.dataset_label}{file_id}.{suffix}"

    #: The prefix where all files are stored, as a URI.
    #:
    #: Eg. ``'file:///my/dataset/collections'``
    #:
    #: (used if dataset_location is generated)
    collection_prefix: Optional[str] = None

    #: The region portion of dataset_folder
    #:
    #: By default, it will split the region code in half
    #:
    #: Eg. ``'012/094'``
    #:
    region_folder = LazyRegionOffset()

    #: The time portion of dataset_folder
    #:
    #: By default, it will be ``"%Y/%m/%d"`` of the dataset's ``'datetime'`` property.
    #:
    #: Eg. ``'2019/03/12'``
    #:
    time_folder = LazyTimeOffset()

    #: The full folder offset from the collection_prefix.
    #:
    #: Example: ``'ga_ls8c_ones_3/090/084/2016/01/21'``
    #:
    #: (used if dataset_location is generated)
    dataset_folder: str = LazyDestinationFolder()

    #: The full uri of the dataset as indexed into ODC.
    #:
    #: **All inner document paths are relative to this.**
    #:
    #: Eg. ``s3://dea-public-data/ga_ls_fc_3/2-5-0/091/086/2020/04/04/ga_ls_fc_091086_2020-04-04.odc-metadata.yaml``
    #:
    #: (Defaults to the metadata path inside the dataset_folder)
    dataset_location: str = LazyDatasetLocation()

    #: The path or URL to the ODC metadata file.
    #:
    #: (if relative, it's relative to self.dataset_location ... but could be absolute too)
    #:
    #: Example: ``'ga_ls8c_ones_3-0-0_090084_2016-01-21_final.odc-metadata.yaml'``
    metadata_file: str = LazyFileName("", "odc-metadata.yaml")

    #: The name of a checksum file
    checksum_file: str = LazyFileName("", "sha1")

    def __init__(
        self,
        properties: Mapping,
        base_product_uri: str = None,
        required_fields: Sequence[str] = (),
        dataset_separator_field: Optional[str] = None,
        allow_unknown_abbreviations: bool = True,
    ) -> None:
        #: The default base URI used in product URI generation
        #:
        #: Example: ``https://collections.dea.ga.gov.au/``
        self.base_product_uri = base_product_uri

        self.required_fields = self._ABSOLUTE_MINIMAL_PROPERTIES.union(required_fields)

        # An extra folder to put each dataset inside, using the value of the given property name.
        self.dataset_separator_field = dataset_separator_field

        if self.dataset_separator_field is not None:
            self.required_fields.add(dataset_separator_field)

        self.allow_unknown_abbreviations = allow_unknown_abbreviations
        #: The underlying dataset properties used for generation.
        self.metadata: Eo3Interface = EnforceRequirementProperties(
            properties, self.required_fields
        )

    @property
    def displayed_collection_number(self) -> Optional[int]:
        # An explicit collection number trumps all.
        if self.metadata.collection_number is not None:
            return int(self.metadata.collection_number)

        # Otherwise it's the first digit of the dataset version.
        if self.metadata.dataset_version is None:
            return None
        return int(self.metadata.dataset_version.split(".")[0])

    def metadata_filename(self, kind: str = "", suffix: str = "yaml") -> str:
        return self.filename(kind, suffix)

    def measurement_filename(
        self, measurement_name: str, suffix: str = "tif", file_id: str = None
    ) -> str:
        """
        Generate the path to a measurement for the current naming conventions.:::

            >> p.names.measurement_file('blue', 'tif')
            'ga_ls8c_ones_3-0-0_090084_2016-01-21_final_blue.tif'

        This is the filename inside the self.dataset_folder
        """
        name = measurement_name.replace(":", "_")

        return self.filename(
            # We use 'band01'/etc in the filename if provided, rather than 'red'
            file_id or name,
            suffix,
        )

    def filename(self, file_id: str, suffix: str) -> str:
        """
        Make a file name according to the current naming conventions' file pattern.

        All filenames have a file_id (eg. "odc-metadata" or "") and a suffix (eg. "yaml")

        Returned file paths are expected to be relative to the ``self.dataset_location``
        """
        file_id = "_" + file_id.replace("_", "-") if file_id else ""
        return self.filename_pattern.format(file_id=file_id, suffix=suffix, n=self)

    def thumbnail_filename(self, kind: str = None, suffix: str = "jpg") -> str:
        """
        Get a thumbnail file path (optionally with the given kind and/or suffix.)
        """
        if kind:
            name = f"{kind}_thumbnail"
        else:
            name = "thumbnail"
        return self.filename(name, suffix)

    def resolve_file(self, path: Location) -> str:
        """
        Convert the given file offset to a fully qualified URL within the dataset location.
        """
        if isinstance(path, Path):
            if path.is_absolute():
                return path.as_uri()
            path = path.as_posix()

        location = self.dataset_location
        resolved = dc_uris.uri_resolve(location, path)
        return resolved

    def resolve_path(self, path: Location) -> Path:
        """
        Convert the given file offset (inside the dataset location) to a Path on the
        local filesystem (if possible).

        :raises ValueError: if the current dataset is not in a file:// location.
        """
        return _as_path(self.resolve_file(path))

    @property
    def dataset_path(self) -> Optional[Path]:
        """
        Get the dataset location as a Path on the local filesystem, if possible.

        :raises ValueError: if the current dataset is not in a file:// location.
        """
        return _as_path(self.dataset_location)

    @property
    def collection_path(self) -> Optional[Path]:
        """
        Get the collection prefix as a Path on the local filesystem, if possible.
        """
        if not self.collection_prefix:
            return None
        try:
            return _as_path(self.collection_prefix)
        except ValueError:
            return None

    def __repr__(self) -> str:
        ps = {"collection_prefix": self.collection_prefix}
        try:
            ps["dataset_location"] = self.dataset_location
        except ValueError:
            ...
        try:
            ps["metadata_file"] = self.metadata_file
        except ValueError:
            ...

        ps = ", ".join(f"{k}={v!r}" for k, v in ps.items())
        return f"{self.__class__.__name__}({ps})"


class DEANamingConventions(NamingConventions):
    """
    Example file structure (note version number in file):

            ga_ls8c_ones_3/090/084/2016/01/21/ga_ls8c_ones_3-0-0_090084_2016-01-21_final.odc-metadata.yaml
    """

    def __init__(
        self,
        properties: Mapping,
        required_fields=(
            "eo:platform",
            "eo:instrument",
            "odc:processing_datetime",
            "odc:producer",
            "odc:product_family",
            "odc:region_code",
            "odc:dataset_version",
        ),
        dataset_separator_field: Optional[str] = None,
    ) -> None:
        # DEA wants consistency via the naming-conventions doc.
        allow_unknown_abbreviations = False
        super().__init__(
            properties,
            DEA_URI_PREFIX,
            required_fields,
            dataset_separator_field,
            allow_unknown_abbreviations,
        )

    product_name: str = LazyProductName(include_collection=True)
    # Stricter: only allow pre-approved abbreviations.
    platform_abbreviated: str = LazyPlatformAbbreviation(
        allow_unknown_abbreviations=False
    )


class DEAS2NamingConventions(DEANamingConventions):
    """
    DEA naming conventions, but with an extra subfolder for each unique datatake.

    It will figure out the datatake if you set a sentinel_tile_id or datastrip_id.
    """

    def __init__(
        self,
        properties: Mapping,
        required_fields=(
            "eo:instrument",
            "eo:platform",
            "odc:dataset_version",
            "odc:processing_datetime",
            "odc:producer",
            "odc:product_family",
            "odc:region_code",
        ),
        dataset_separator_field="sentinel:datatake_start_datetime",
    ) -> None:
        super().__init__(
            properties, required_fields, dataset_separator_field=dataset_separator_field
        )


class DEADerivativesNamingConventions(DEANamingConventions):
    """
    Common derived products.

    Unlike plain 'DEA', they use an explicit collection number (odc:collection_number)
    in the product name which may differ from the software's dataset version
    (odc:dataset_version)

    Example file structure (note version number in folder):

        ga_ls_wo_3/1-6-0/090/081/1998/07/30/ga_ls_wo_3_090081_1998-07-30_interim.odc-metadata.yaml

    Derivatives have a slightly different folder structure.

    And they only show constellations (eg. "ls_" or "s2_") rather than the specific
    satellites in their names (eg. "ls8_").

    They have a version-number folder instead of putting it in each filename.

    And version numbers may not match the collection number (`odc:collection_number` is
    mandatory).
    """

    def __init__(
        self,
        properties: Mapping,
        required_fields: Sequence[str] = (
            "eo:platform",
            "odc:dataset_version",
            "odc:collection_number",
            "odc:processing_datetime",
            "odc:producer",
            "odc:product_family",
            "odc:region_code",
            "dea:dataset_maturity",
        ),
        dataset_separator_field: Optional[str] = None,
    ) -> None:
        super().__init__(
            properties,
            required_fields=required_fields,
            dataset_separator_field=dataset_separator_field,
        )

    # Derivates put the version in the folder instead.
    dataset_label = LazyLabel(include_version=False)
    product_name = LazyProductName(include_instrument=False, include_collection=True)

    dataset_folder = LazyDestinationFolder(
        include_version=True,
        include_non_final_maturity=False,
    )

    platform_abbreviated = LazyPlatformAbbreviation(
        show_specific_platform=False,
        allow_unknown_abbreviations=False,
    )


class DEAS2DerivativesNamingConventions(DEADerivativesNamingConventions):
    """
    Sentinel-2-based DEA derivative naming conventions. Unlike regular
    derivatives, there's an extra subfolder for each unique datatake.

    It will figure out the datatake if you set a sentinel_tile_id
    or datastrip_id.
    """

    def __init__(self, properties: Mapping) -> None:
        super().__init__(
            properties,
            dataset_separator_field="sentinel:datatake_start_datetime",
        )


class AfricaProductName:
    def __get__(self, c: "NamingConventions", owner) -> str:
        if c.metadata.product_name:
            return c.metadata.product_name
        return f"{c.metadata.product_family}_{c.platform_abbreviated}"


class DEAfricaNamingConventions(NamingConventions):
    """
    DEAfrica avoids org names and uses simpler "{family}_{platform}" product names.

    Eg. "wo_ls" (water observations of landsat)

    """

    product_name = AfricaProductName()
    dataset_label = LazyLabel(include_version=False)
    dataset_folder = LazyDestinationFolder(
        include_version=True,
        include_non_final_maturity=False,
    )
    platform_abbreviated = LazyPlatformAbbreviation(
        show_specific_platform=False,
        allow_unknown_abbreviations=False,
    )

    def __init__(
        self,
        properties: Mapping,
    ) -> None:
        super().__init__(
            properties,
            base_product_uri="https://digitalearthafrica.org",
            required_fields=(
                "eo:platform",
                "odc:producer",
                "odc:region_code",
                "odc:product_family",
                "odc:dataset_version",
            ),
        )


KNOWN_CONVENTIONS = dict(
    default=NamingConventions,
    dea=DEANamingConventions,
    dea_s2=DEAS2NamingConventions,
    dea_s2_derivative=DEAS2DerivativesNamingConventions,
    dea_c3=DEADerivativesNamingConventions,
    deafrica=DEAfricaNamingConventions,
)


def namer(
    properties: Union[Eo3Dict, Eo3Interface, dict] = None,
    *,
    collection_prefix: Location = None,
    conventions: str = "default",
) -> "NamingConventions":
    """
    Create a naming instance of the given conventions.

    Conventions: 'default', 'dea', 'deafrica', ...

    You usually give it existing properties, but you can use the return value's
    :attr:`.metadata <eodatasets3.NamingConventions.metadata>` field to set properties afterwards.

    """
    if conventions not in KNOWN_CONVENTIONS:
        available = ", ".join(KNOWN_CONVENTIONS.keys())
        raise ValueError(
            f"Unknown naming conventions: {conventions}. Possibilities: {available}"
        )
    if isinstance(properties, Eo3Interface):
        properties = properties.properties
    if properties is None:
        properties = Eo3Dict()
    conventions = KNOWN_CONVENTIONS[conventions](properties)
    if collection_prefix:
        conventions.collection_prefix = resolve_location(collection_prefix)

    return conventions
