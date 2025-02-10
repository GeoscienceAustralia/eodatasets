#!/usr/bin/env python
"""
Repackage a USGS Collection-1 tar for faster read access.

They arrive as a *.tar.gz with inner uncompressed tiffs, which Josh's tests have found to be too slow to read.

We compress the inner tiffs and store them in an uncompressed tar. This allows random reads within the files.
We also append a checksum file at the end of the tar.
"""

import copy
import io
import socket
import stat
import sys
import tarfile
import tempfile
import traceback
from collections.abc import Callable, Iterable
from contextlib import suppress
from functools import partial
from itertools import chain
from pathlib import Path
from typing import IO

import click
import numpy
import rasterio
import structlog
from structlog.processors import (
    JSONRenderer,
    StackInfoRenderer,
    TimeStamper,
    format_exc_info,
)

from eodatasets3.ui import PathPath
from eodatasets3.verify import PackageChecksum

_PREDICTOR_TABLE = {
    "int8": 2,
    "uint8": 2,
    "int16": 2,
    "uint16": 2,
    "int32": 2,
    "uint32": 2,
    "int64": 2,
    "uint64": 2,
    "float32": 3,
    "float64": 3,
}

# The info of a file, and a method to open the file for reading.
ReadableMember = tuple[tarfile.TarInfo, Callable[[], IO]]

_LOG = structlog.get_logger()


class RecompressFailure(Exception):
    pass


def _create_tarinfo(path: Path, name=None) -> tarfile.TarInfo:
    """
    Create a TarInfo ("tar member") based on the given filesystem path.

    (these contain the information of a file, such as permissions, when writing to a tar file)

    This code is based on TarFile.gettarinfo(), but doesn't need an existing tar file.
    """
    # We're avoiding convenience methods like `path.is_file()`, to minimise repeated `stat()` calls on lustre.
    s = path.stat()
    info = tarfile.TarInfo(name or path.name)

    if stat.S_ISREG(s.st_mode):
        info.size = s.st_size
        info.type = tarfile.REGTYPE
    elif stat.S_ISDIR(s.st_mode):
        info.type = tarfile.DIRTYPE
        info.size = 0
    else:
        raise NotImplementedError(
            f"Only regular files and directories are supported for extracted datasets. "
            f"({path.name} in {path.absolute().parent})"
        )

    info.mode = s.st_mode
    info.uid = s.st_uid
    info.gid = s.st_gid
    info.mtime = s.st_mtime

    if tarfile.pwd:
        try:
            info.uname = tarfile.pwd.getpwuid(info.uid)[0]
        except KeyError:
            pass
    if tarfile.grp:
        try:
            info.gname = tarfile.grp.getgrgid(info.gid)[0]
        except KeyError:
            pass
    return info


def _tar_members(in_tar: tarfile.TarFile) -> Iterable[ReadableMember]:
    """Get readable files (members) from a tar"""
    members: list[tarfile.TarInfo] = in_tar.getmembers()

    for member in members:
        # We return a lambda/callable so that the file isn't opened until it's needed.
        yield member, partial(in_tar.extractfile, member)


def _folder_members(path: Path, base_path: Path = None) -> Iterable[ReadableMember]:
    """
    Get readable files (presented as tar members) from a directory.
    """
    if not base_path:
        base_path = path

    # Note that the members in input USGS tars are sorted alphabetically.
    # We'll sort our own inputs to match.
    # (The primary practical benefit is predictable outputs in tests)

    for item in sorted(path.iterdir()):
        member = _create_tarinfo(item, name=str(item.relative_to(base_path)))
        if member.type == tarfile.DIRTYPE:
            yield member, None
            yield from _folder_members(item, base_path=path)
        else:
            # We return a lambda/callable so that the file isn't opened until it's needed.
            yield member, partial(item.open, "rb")


def repackage_tar(
    input_path: Path,
    input_files: Iterable[ReadableMember],
    output_tar_path: Path,
    clean_inputs: bool,
    **compress_args,
) -> bool:
    log = _LOG.bind(
        name=output_tar_path.stem,
        in_path=str(input_path.absolute()),
        out_path=str(output_tar_path.absolute()),
    )

    if output_tar_path.exists():
        log.info("skip.exists")
        return True

    try:
        members = list(input_files)

        # Add the MTL file to the beginning of the output tar, so it can be accessed faster.
        # This slows down this repackage a little, as we're seeking/decompressing the input stream an extra time.
        _reorder_tar_members(members, input_path.name)

        _create_tar_with_files(input_path, members, output_tar_path, **compress_args)

        log.info(
            "complete",
            in_size=sum(m.size for m, _ in members),
            in_count=len(members),
            # The user/group give us a hint as to whether this was repackaged outside of USGS.
            in_users=list({(member.uname, member.gname) for member, _ in members}),
            out_size=output_tar_path.stat().st_size,
        )

        result_exists = output_tar_path.exists()
        if not result_exists:
            # This should never happen, so it's an exception.
            raise RuntimeError(f"No output after a success? Expected {output_tar_path}")

        if clean_inputs:
            log.info("input.cleanup")
            please_remove(input_path, excluding=output_tar_path)
    except Exception:
        log.exception("error", exc_info=True)  # noqa: G202
        return False
    return True


def _create_tar_with_files(
    input_path: Path,
    members: list[ReadableMember],
    output_tar_path: Path,
    **compress_args,
) -> None:
    """
    Package and compress the given input files to a new tar path.

    The output tar path is written atomically, so on failure it will only exist if complete.
    """

    out_dir: Path = output_tar_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    verify = PackageChecksum()

    # Use a temporary file so that we can move to the output path atomically.
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=str(out_dir)) as tmpdir:
        tmpdir = Path(tmpdir).absolute()
        tmp_out_tar = tmpdir.joinpath(output_tar_path.name)

        with tarfile.open(tmp_out_tar, "w") as out_tar:
            with click.progressbar(
                label=input_path.name,
                length=sum(member.size for member, _ in members),
                file=sys.stderr,
            ) as progress:
                file_number = 0
                for readable_member in members:
                    file_number += 1
                    progress.label = (
                        f"{input_path.name} ({file_number:2d}/{len(members)})"
                    )

                    _recompress_tar_member(
                        readable_member, out_tar, compress_args, verify, tmpdir
                    )

                    member, _ = readable_member
                    progress.update(member.size)

            # Append sha1 checksum file
            checksum_path = tmpdir / "package.sha1"
            verify.write(checksum_path)
            checksum_path.chmod(0o664)
            out_tar.add(checksum_path, checksum_path.name)

            # Match the lower r/w permission bits to the output folder.
            # (Temp directories default to 700 otherwise.)
            tmp_out_tar.chmod(out_dir.stat().st_mode & 0o777)
            # Our output tar is complete. Move it into place.
            tmp_out_tar.rename(output_tar_path)


def _recompress_tar_member(
    readable_member: ReadableMember,
    out_tar: tarfile.TarFile,
    compress_args: dict,
    verify: PackageChecksum,
    tmpdir: Path,
):
    member, open_member = readable_member

    new_member = copy.copy(member)
    # Copy with a minimum 664 permission, which is used by USGS tars.
    # (some of our repacked datasets have only user read permission.)
    new_member.mode = new_member.mode | 0o664

    # If it's a tif, check whether it's compressed.
    if member.name.lower().endswith(".tif"):
        with open_member() as input_fp, rasterio.open(input_fp) as ds:
            if not ds.profile.get("compress"):
                # No compression: let's compress it
                with rasterio.MemoryFile(filename=member.name) as memory_file:
                    try:
                        _recompress_image(ds, memory_file, **compress_args)
                    except Exception as e:
                        raise RecompressFailure(f"Error during {member.name}") from e
                    new_member.size = memory_file.getbuffer().nbytes
                    out_tar.addfile(new_member, memory_file)

                    # Image has been written. Seek to beginning to take a checksum.
                    memory_file.seek(0)
                    verify.add(memory_file, tmpdir / new_member.name)
                    return
            else:
                # It's already compressed, we'll fall through and copy it verbatim.
                pass

    if member.size == 0:
        # Typically a directory entry.
        out_tar.addfile(new_member)
        return

    # Copy unchanged into target (typically text/metadata files).
    with open_member() as member:
        file_contents = member.read()
    out_tar.addfile(new_member, io.BytesIO(file_contents))
    verify.add(io.BytesIO(file_contents), tmpdir / new_member.name)
    del file_contents


def _reorder_tar_members(members: list[ReadableMember], identifier: str):
    """
    Put the (tiny) MTL file at the beginning of the tar so that it's always quick to read.
    """
    # Find MTL
    for i, (member, _) in enumerate(members):
        if "_MTL" in member.path:
            mtl_index = i
            break
    else:
        formatted_members = "\n\t".join(m.name for m, _ in members)
        raise ValueError(
            f"No MTL file found in package {identifier}. Have:\n\t{formatted_members}"
        )

    # Move to front
    mtl_item = members.pop(mtl_index)
    members.insert(0, mtl_item)


def _recompress_image(
    input_image: rasterio.DatasetReader,
    output_fp: rasterio.MemoryFile,
    zlevel=9,
    block_size=(512, 512),
):
    """
    Read an image from given file pointer, and write as a compressed GeoTIFF.
    """
    # noinspection PyUnusedLocal

    block_size_y, block_size_x = block_size

    if len(input_image.indexes) != 1:
        raise ValueError(
            f"Expecting one-band-per-tif input (USGS packages). "
            f"Input has multiple layers {input_image.indexes!r}"
        )

    array: numpy.ndarray = input_image.read(1)
    profile = input_image.profile
    profile.update(
        driver="GTiff",
        predictor=_PREDICTOR_TABLE[array.dtype.name],
        compress="deflate",
        zlevel=zlevel,
        blockxsize=block_size_x,
        blockysize=block_size_y,
        tiled=True,
    )

    with output_fp.open(**profile) as output_dataset:
        output_dataset.write(array, 1)
        # Copy gdal metadata
        output_dataset.update_tags(**input_image.tags())
        output_dataset.update_tags(1, **input_image.tags(1))


@click.command(help=__doc__)
@click.option(
    "--output-base",
    type=PathPath(file_okay=False, writable=True),
    help="The base output directory (default to same dir as input if --clean-inputs).",
)
@click.option(
    "--zlevel", type=click.IntRange(0, 9), default=5, help="Deflate compression level."
)
@click.option(
    "--block-size", type=int, default=512, help="Compression block size (both x and y)"
)
@click.option(
    "--clean-inputs/--no-clean-inputs",
    default=False,
    help="Delete originals after repackaging",
)
@click.option("-f", "input_file", help="Read paths from file", type=click.File("r"))
@click.argument("paths", nargs=-1, type=PathPath(exists=True, readable=True))
def main(
    paths: list[Path],
    input_file,
    output_base: Path,
    zlevel: int,
    clean_inputs: bool,
    block_size: int,
):
    # Structured (json) logging goes to stdout
    structlog.configure(
        processors=[
            StackInfoRenderer(),
            format_exc_info,
            TimeStamper(utc=False, fmt="iso"),
            JSONRenderer(),
        ]
    )

    if (not output_base) and (not clean_inputs):
        raise click.UsageError(
            "Need to specify either a different output directory (--output-base) "
            "or to clean inputs (--clean-inputs)"
        )

    if input_file:
        paths = chain((Path(p.strip()) for p in input_file), paths)

    with rasterio.Env():
        total = failures = 0
        for path in paths:
            total += 1

            # Input is either a tar.gz file, or a directory containing an MTL (already extracted)
            if path.suffix.lower() == ".gz":
                with tarfile.open(str(path), "r") as in_tar:
                    success = repackage_tar(
                        path,
                        _tar_members(in_tar),
                        _output_tar_path(output_base, path),
                        clean_inputs=clean_inputs,
                        zlevel=zlevel,
                        block_size=(block_size, block_size),
                    )

            elif path.is_dir():
                success = repackage_tar(
                    path,
                    _folder_members(path),
                    _output_tar_path_from_directory(output_base, path),
                    clean_inputs=clean_inputs,
                    zlevel=zlevel,
                    block_size=(block_size, block_size),
                )
            else:
                raise ValueError(
                    f"Expected either tar.gz or a dataset folder. Got: {path!r}"
                )

            if not success:
                failures += 1
    if total > 1:
        _LOG.info(
            "node.finish",
            host=socket.getfqdn(),
            total_count=total,
            failure_count=failures,
        )
    sys.exit(failures)


def please_remove(path: Path, excluding: Path):
    """
    Delete all of path, excluding the given path.
    """
    if path.absolute() == excluding.absolute():
        return

    if path.is_dir():
        for p in path.iterdir():
            please_remove(p, excluding)
        with suppress(OSError):
            path.rmdir()
    else:
        path.unlink()


def _format_exception(e: BaseException):
    """
    Shamelessly stolen from stdlib's logging module.
    """
    with io.StringIO() as sio:
        traceback.print_exception(e.__class__, e, e.__traceback__, None, sio)
        return sio.getvalue().strip()


def _output_tar_path(base_output, input_path):
    if base_output:
        out_path = _calculate_out_base_path(base_output, input_path)
    else:
        out_path = input_path

    # Remove .gz suffix
    name = out_path.stem
    if not name.endswith(".tar"):
        raise RuntimeError(f"Expected path to end in .tar.gz, got: {out_path}")
    return out_path.with_name(name)


def _output_tar_path_from_directory(base_output, input_path):
    mtl_files = list(input_path.glob("*_MTL.txt"))
    if not mtl_files:
        raise ValueError(f"Dataset has no mtl: {input_path}")
    if len(mtl_files) > 1:
        _LOG.warning("multiple.mtl.files", in_path=input_path)
    mtl_file = mtl_files[0]
    dataset_name = mtl_file.name.replace("_MTL.txt", "")

    if base_output:
        return _calculate_out_base_path(base_output, input_path) / f"{dataset_name}.tar"
    else:
        return input_path / f"{dataset_name}.tar"


def _calculate_out_base_path(out_base: Path, path: Path) -> Path:
    if "USGS" not in path.parts:
        raise ValueError(
            "Expected AODH input path structure, "
            "eg: /AODH/USGS/L1/Landsat/C1/092_091/LT50920911991126/LT05_L1GS_092091_19910506_20170126_01_T2.tar.gz"
        )
    # The directory structure after the "USGS" folder is recreated onto the output base folder.
    return out_base.joinpath(*path.parts[path.parts.index("USGS") + 1 : -1], path.name)


if __name__ == "__main__":
    main()
