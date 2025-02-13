import re
import sqlite3
from pathlib import Path
from pprint import pprint
from typing import Dict, Iterable, List, Optional, Tuple

import click
from attr import define
from click import echo, pass_obj

from eodatasets3.ui import PathPath

NCI_L1C_LOCATION = Path("/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C")

DEFAULT_DB = Path(__file__).parent / "s2_regions.db"


_AREA_PARTS = re.compile(r"(\d+)([NS])(\d+)([EW])")


def area_to_tuple(area: str) -> Tuple[int, int, int, int]:
    """
    >>> area_to_tuple('20S120E-25S125E')
    (-20, 120, -25, 125)
    >>> area_to_tuple('00N160W-05S155W')
    (0, -160, -5, -155)
    """

    def point(part: str) -> Tuple[int, int]:
        match = _AREA_PARTS.match(part)
        if match is None:
            raise ValueError(f"Not an area? {part!r} to {_AREA_PARTS!r}")
        lat, ns, lon, ew = match.groups()
        if ns == "S":
            lat = "-" + lat
        if ew == "W":
            lon = "-" + lon

        return int(lat), int(lon)

    first, second = area.split("-")
    return (*point(first), *point(second))


@define
class FolderInfo:
    """
    Information extracted from a standard S2 folder layout.
    """

    year: int
    month: int
    area: str
    region_code: Optional[str]

    # Compiled regexp for extracting year, month and region
    # Standard layout is of the form: 'L1C/{yyyy}/{yyyy}-{mm}/{area}/S2*_{region}_{timestamp}(.zip)'
    STANDARD_SUBFOLDER_LAYOUT = re.compile(
        r"(\d{4})/(\d{4})-(\d{2})/([\dNESW]+-[\dNESW]+)/"
        r"S2[ABC](?:_OPER_PRD)?_MSIL1C(?:_PDMC)?(?:_[a-zA-Z0-9]+){3}(?:_T([A-Z\d]+))?_[\dT]+(\.zip|/tileInfo\.json)?$"
    )

    @property
    def area_tuple(self):
        return area_to_tuple(self.area)

    @classmethod
    def for_path(cls, path: Path) -> Optional["FolderInfo"]:
        """
        Can we extract information from the given path?

        Returns None if we can't.
        """
        m = cls.STANDARD_SUBFOLDER_LAYOUT.search(path.as_posix())
        if not m:
            return None

        year, year2, month, area, region_code, extension = m.groups()
        if year != year2:
            raise ValueError(f"Year mismatch in {path}")

        return FolderInfo(int(year), int(month, 10), area, region_code)


class RegionLookup:
    """
    A lookup table for S2 regions and areas.

    This is for matching areas to region_codes on older S2 data that does not contain
    a region_code.

    The table is stored as an sqlite database.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path
        self._db: Optional[sqlite3.Connection] = None

    def open(self):
        if self._db is None:
            if self.db_path is None:
                # Open the default db read-only
                self._db = sqlite3.connect(f"{DEFAULT_DB.as_uri()}?mode=ro", uri=True)
            else:
                self._db = sqlite3.connect(self.db_path)

    def close(self):
        if self._db:
            self._db.close()
            self._db = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def add_from_file(self, paths_file: Path) -> int:
        """Add a paths to the lookup table"""
        opener = open
        if paths_file.suffix.lower() == ".gz":
            import gzip

            opener = gzip.open

        return self.add_from_paths(Path(p.strip()) for p in opener(paths_file, "rt"))

    def add_from_scanning_path(self, l1cs: Path) -> int:
        """Scan for zips in a directory and add them to the table"""
        return self.add_from_paths(l1cs.rglob("*.zip"))

    def add_from_paths(self, paths: Iterable[Path]):
        """Add the given pathsto the lookup table"""
        self.open()
        s = self._db.cursor()
        s.execute(
            """
            create table if not exists regions (
                lat1 integer not null,
                lon1 integer not null,
                lat2 integer not null,
                lon2 integer not null,
                region_code text null,
                primary key (lat1, lon1, lat2, lon2, region_code)
            )
        """
        )
        s.execute("begin")

        def vals(path: Path):
            i = FolderInfo.for_path(path)
            return (
                *i.area_tuple,
                i.region_code,
            )

        res = s.executemany(
            "insert into regions values (?,?,?,?,?) on conflict do nothing",
            (vals(path) for path in paths),
        )
        insert_count = res.rowcount
        s.execute("commit")
        return insert_count

    def get(self, area: str) -> List[str]:
        """Get known region codes for an area"""
        self.open()

        def get_lazy():
            for [code] in self._db.execute(
                """
                    select distinct region_code from regions where (lat1, lon1, lat2, lon2)  = (?, ?, ?, ?)
                """,
                area_to_tuple(area),
            ):
                yield code

        return list(get_lazy())

    def vacuum(self):
        """Vacuum the database (reclaim space)"""
        self.open()
        self._db.execute("vacuum")

    def stats(self) -> Dict:
        """Get stats about the lookup table"""
        self.open()
        res = self._db.execute(
            """
            select
                count(*) as total,
                count(distinct region_code) as unique_regions
                --- count(distinct (lat1, lon1, lat2, lon2)) as unique_areas
            from regions
        """
        ).fetchone()

        return res


@click.group("s2_regions", help=__doc__)
@click.option("--db", default=DEFAULT_DB, type=PathPath())
@click.pass_context
def cli(ctx, db: Path):
    ctx.obj = RegionLookup(db)


@cli.command("create", help="Recreate the database")
@click.option("--scan-path", default=None, type=PathPath(exists=True))
@click.option("-f", "paths_file", default=None, type=PathPath(exists=True))
@pass_obj
def cli_create(db: RegionLookup, scan_path: Path, paths_file: Path):
    if scan_path is None and paths_file is None:
        echo("Nothing specified. Scanning default NCI location.")
        scan_path = NCI_L1C_LOCATION

    if scan_path is not None:
        echo("Scanning....")
        inserted = db.add_from_scanning_path(scan_path)
        echo(f"Found {inserted} items in {scan_path}")

    if paths_file is not None:
        echo("Adding from file....")
        inserted = db.add_from_file(paths_file)
        echo(f"Loaded {inserted} items from file {paths_file}")

    db.vacuum()
    db.close()


@cli.command("get", help="Get region codes for an area")
@click.argument("area", type=str)
@click.pass_obj
def cli_get(db: RegionLookup, area: str):
    for region in db.get(area):
        echo(region)

    db.close()


@cli.command("info", help="Get region codes for an area")
@click.pass_obj
def cli_info(
    db: RegionLookup,
):
    pprint(db.stats())
    db.close()


if __name__ == "__main__":
    cli()
