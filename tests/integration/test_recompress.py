import shutil
import tarfile
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from eodatasets3 import verify
from eodatasets3.scripts import recompress

this_folder = Path(__file__).parent
packaged_base: Path = this_folder / "data/recompress_packed"
packaged_offset = "USGS/L1/Landsat/C1/092_091/LT50920911991126/LT05_L1GS_092091_19910506_20170126_01_T2.tar.gz"
packaged_path = packaged_base / packaged_offset
unpackaged_base: Path = this_folder / "data/recompress_unpackaged"
unpackaged_offset = "USGS/L1/Landsat/C1/092_091/LT50920911991126"
unpackaged_path = unpackaged_base / unpackaged_offset


def please_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(str(src), str(dst))
    else:
        shutil.copy(str(src), str(dst))


@pytest.mark.parametrize(
    "base_in_path,in_offset",
    [(packaged_base, packaged_offset), (unpackaged_base, unpackaged_offset)],
    ids=("packaged", "unpackaged"),
)
def test_recompress_dataset(base_in_path: Path, in_offset: str, tmp_path: Path):
    test_dataset = base_in_path / in_offset
    assert test_dataset.exists()

    # Copy our input into the temp directory, as we'll change it.
    input_path: Path = tmp_path / in_offset

    please_copy(test_dataset, input_path)

    assert input_path.exists()

    # Same folder as the input!
    output_base = tmp_path / "USGS"

    _run_recompress(input_path, "--clean-inputs")

    # If input was a file, it should no longer exist.
    # (a directory will still exist it contains the output [which is checked below])
    assert not input_path.is_file(), "Input file was not cleaned up"

    expected_output = (
        output_base
        / "L1/Landsat/C1/092_091/LT50920911991126"
        / "LT05_L1GS_092091_19910506_20170126_01_T2.tar"
    )

    # Pytest has better error messages for strings than Paths.
    all_output_files = {
        str(p.relative_to(output_base)) for p in output_base.rglob("*") if p.is_file()
    }

    assert len(all_output_files) == 1, (
        f"Expected one output tar file. Got: {len(all_output_files)}"
        f"\n\t" + "\n\t".join(all_output_files)
    )
    assert all_output_files == {str(expected_output.relative_to(output_base))}

    assert expected_output.exists(), (
        f"No output produced in expected location {expected_output}."
    )

    # It should contain all of our files
    checksums, members = _get_checksums_members(expected_output)

    member_names = [m.name for m in members]

    # Note that MTL is first. We do this deliberately so it's quick to access.
    # The others are alphabetical, as with USGS tars.
    # (Not that it matters, but reprocessing stability is nice.)
    assert member_names == [
        "LT05_L1GS_092091_19910506_20170126_01_T2_MTL.txt",
        "LT05_L1GS_092091_19910506_20170126_01_T2_ANG.txt",
        "LT05_L1GS_092091_19910506_20170126_01_T2_B1.TIF",
        "LT05_L1GS_092091_19910506_20170126_01_T2_B2.TIF",
        "LT05_L1GS_092091_19910506_20170126_01_T2_B3.TIF",
        "LT05_L1GS_092091_19910506_20170126_01_T2_B4.TIF",
        "LT05_L1GS_092091_19910506_20170126_01_T2_B5.TIF",
        "LT05_L1GS_092091_19910506_20170126_01_T2_B6.TIF",
        "LT05_L1GS_092091_19910506_20170126_01_T2_B7.TIF",
        "LT05_L1GS_092091_19910506_20170126_01_T2_BQA.TIF",
        "README.GTF",
        "extras",
        "extras/example-file.txt",
        "package.sha1",
    ]

    member_sizes = {m.name: m.size for m in members}

    # Text files should be unchanged.
    assert member_sizes["LT05_L1GS_092091_19910506_20170126_01_T2_MTL.txt"] == 6693

    assert "LT05_L1GS_092091_19910506_20170126_01_T2_MTL.txt" in checksums, "No MTL?"
    assert (
        checksums["LT05_L1GS_092091_19910506_20170126_01_T2_MTL.txt"]
        == "beb4d546dc5e2850b2f33384bfbc6cf15b724197"
    )

    # Are they the expected number of bytes?
    assert member_sizes["package.sha1"] == 1010
    assert member_sizes["README.GTF"] == 8686
    assert member_sizes["LT05_L1GS_092091_19910506_20170126_01_T2_ANG.txt"] == 34884


def test_recompress_gap_mask_dataset(tmp_path: Path):
    input_path = this_folder.joinpath(
        "data/recompress_packed/USGS/L1/Landsat/C1/091_080/LE70910802008014",
        "LE07_L1GT_091080_20080114_20161231_01_T2.tar.gz",
    )
    assert input_path.exists()

    output_base = tmp_path / "out"

    with ExpectPathUnchanged(input_path):
        _run_recompress(input_path, "--output-base", str(output_base))

    expected_output = (
        output_base
        / "L1/Landsat/C1/091_080/LE70910802008014"
        / "LE07_L1GT_091080_20080114_20161231_01_T2.tar"
    )

    # Pytest has better error messages for strings than Paths.
    all_output_files = [str(p) for p in output_base.rglob("*") if p.is_file()]

    assert len(all_output_files) == 1, (
        "Expected one output tar file. Got: \n\t" + "\n\t".join(all_output_files)
    )
    assert all_output_files == [str(expected_output)]

    assert expected_output.exists(), (
        f"No output produced in expected location {expected_output}."
    )

    # It should contain all of our files
    checksums, members = _get_checksums_members(expected_output)

    member_names = [(m.name, f"{m.mode:o}") for m in members]

    # Note that MTL is first. We do this deliberately so it's quick to access.
    # The others are alphabetical, as with USGS tars.
    # (Not that it matters, but reprocessing stability is nice.)
    assert member_names == [
        ("LE07_L1GT_091080_20080114_20161231_01_T2_MTL.txt", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_ANG.txt", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B1.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B2.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B3.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B4.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B5.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B6_VCID_1.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B6_VCID_2.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B7.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_B8.TIF", "664"),
        ("LE07_L1GT_091080_20080114_20161231_01_T2_BQA.TIF", "664"),
        ("README.GTF", "664"),
        ("gap_mask", "775"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B1.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B2.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B3.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B4.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B5.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B6_VCID_1.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B6_VCID_2.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B7.TIF", "664"),
        ("gap_mask/LE07_L1GT_091080_20080114_20161231_01_T2_GM_B8.TIF", "664"),
        ("package.sha1", "664"),
    ]

    ####
    # If packaging is rerun, the output should not be touched!
    # ie. skip if output exists.
    unchanged_output = ExpectPathUnchanged(
        expected_output, "Output file shouldn't be touched on rerun of compress"
    )
    unchanged_input = ExpectPathUnchanged(
        input_path, "Input path shouldn't be cleaned when output is skipped"
    )
    with unchanged_input, unchanged_output:
        _run_recompress(input_path, "--clean-inputs", "--output-base", str(output_base))


def test_recompress_dirty_dataset(tmp_path: Path):
    # We found some datasets that have been "expanded" and later retarred.
    # They have extra tifs and jpegs created from the bands.
    # The TIFs have compression and multiple bands, unlike USGS tifs.
    # We expect such tifs to be unmodified by this repackager.

    input_path = this_folder.joinpath(
        "data/recompress_packed/USGS/L1/Landsat/C1/091_075/LC80910752016348",
        "LC08_L1TP_091075_20161213_20170316_01_T2.tar.gz",
    )
    assert input_path.exists()

    output_base = tmp_path / "out"

    with ExpectPathUnchanged(input_path):
        _run_recompress(input_path, "--output-base", str(output_base))

    expected_output = (
        output_base
        / "L1/Landsat/C1/091_075/LC80910752016348"
        / "LC08_L1TP_091075_20161213_20170316_01_T2.tar"
    )

    # Pytest has better error messages for strings than Paths.
    all_output_files = [str(p) for p in output_base.rglob("*") if p.is_file()]

    assert len(all_output_files) == 1, (
        "Expected one output tar file. Got: \n\t" + "\n\t".join(all_output_files)
    )
    assert all_output_files == [str(expected_output)]

    assert expected_output.exists(), (
        f"No output produced in expected location {expected_output}."
    )

    checksums, members = _get_checksums_members(expected_output)

    assert (
        checksums["LC08_L1TP_091075_20161213_20170316_01_T2.tif"]
        == "57cafe38c2f4f94cd15a05cfd918911889b8b03f"
    ), "compressed tif has changed. It should be unmodified."

    member_names = [m.name for m in members]
    # Note that MTL is first. We do this deliberately so it's quick to access.
    # The others are alphabetical, as with USGS tars.
    # (Not that it matters, but reprocessing stability is nice.)
    print("\n".join(member_names))
    assert member_names == [
        "LC08_L1TP_091075_20161213_20170316_01_T2_MTL.txt",
        "LC08_L1TP_091075_20161213_20170316_01_T2_ANG.txt",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B10.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B11.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B1.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B2.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B3.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B4.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B5.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B6.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B7.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B8.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_B9.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2_BQA.TIF",
        "LC08_L1TP_091075_20161213_20170316_01_T2.IMD",
        "LC08_L1TP_091075_20161213_20170316_01_T2.jpeg",
        "LC08_L1TP_091075_20161213_20170316_01_T2_QB.jpeg",
        "LC08_L1TP_091075_20161213_20170316_01_T2_QB.tif",
        "LC08_L1TP_091075_20161213_20170316_01_T2.tif",
        "LC08_L1TP_091075_20161213_20170316_01_T2.tif.msk",
        "LC08_L1TP_091075_20161213_20170316_01_T2_TIR.jpeg",
        "LC08_L1TP_091075_20161213_20170316_01_T2_TIR.tif",
        "package.sha1",
    ]


def test_run_with_corrupt_data(tmp_path: Path):
    output_path = tmp_path / "out"
    output_path.mkdir()

    # Recompress expects the dataset in a "USGS" folder structure.
    # Should bail otherwise rather than write data to unexpected locations!
    # (we may expand this in the future, but being safe for now)
    non_usgs_path = tmp_path / packaged_path.name
    non_usgs_path.symlink_to(packaged_path)

    with pytest.raises(ValueError, match="Expected AODH input path structure"):
        _run_recompress(non_usgs_path, "--output-base", str(output_path))


def _run_recompress(input_path: Path, *args, expected_return=0):
    if isinstance(args, str):
        args = [args]

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        res: Result = CliRunner().invoke(
            recompress.main,
            (
                # Out test data is smaller than the default block size.
                "--block-size",
                "32",
                *args,
                str(input_path),
            ),
            catch_exceptions=False,
        )

    if expected_return is not None:
        assert res.exit_code == expected_return, res.output
    return res


def _get_checksums_members(out_tar: Path) -> tuple[dict, list[tarfile.TarInfo]]:
    with tarfile.open(out_tar, "r") as tar:
        members: list[tarfile.TarInfo] = tar.getmembers()

        # Checksum is last (can be calculated while streaming)
        checksum_member = members[-1]
        assert checksum_member.name == "package.sha1"
        checksums = {}
        for line in tar.extractfile(checksum_member).readlines():
            chksum, path = line.decode("utf-8").split("\t")
            path = path.strip()
            assert path not in checksums, f"Path is repeated in checksum file: {path}"
            checksums[path] = chksum
    return checksums, members


def test_calculate_out_path(tmp_path: Path):
    out_base = tmp_path / "out"

    # When input is a tar file, use the same name on output.
    path = Path(
        "/test/in/l1-data/USGS/L1/C1/092_091/LT50920911991126/"
        "LT05_L1GS_092091_19910506_20170126_01_T2.tar.gz"
    )
    assert_path_eq(
        out_base.joinpath(
            "L1/C1/092_091/LT50920911991126/"
            "LT05_L1GS_092091_19910506_20170126_01_T2.tar"
        ),
        recompress._output_tar_path(out_base, path),
    )

    # When no output directory, put it in same folder.
    path = Path(
        "/test/in/l1-data/USGS/L1/C1/092_091/LT50920911991126/"
        "LT05_L1GS_092091_19910506_20170126_01_T2.tar.gz"
    )
    assert_path_eq(
        Path(
            "/test/in/l1-data/USGS/L1/C1/092_091/LT50920911991126/"
            "LT05_L1GS_092091_19910506_20170126_01_T2.tar"
        ),
        recompress._output_tar_path(None, path),
    )

    # When input is a directory, use the MTL file's name for the output.
    path = tmp_path / "USGS/L1/092_091/LT50920911991126"
    path.mkdir(parents=True)
    mtl = path / "LT05_L1GS_092091_19910506_20170126_01_T2_MTL.txt"
    mtl.write_text("fake mtl")
    assert_path_eq(
        out_base.joinpath(
            "L1/092_091/LT50920911991126/LT05_L1GS_092091_19910506_20170126_01_T2.tar"
        ),
        recompress._output_tar_path_from_directory(out_base, path),
    )
    # No output path, it goes inside the folder.
    assert_path_eq(
        path.joinpath("LT05_L1GS_092091_19910506_20170126_01_T2.tar"),
        recompress._output_tar_path_from_directory(None, path),
    )


class ExpectPathUnchanged:
    """
    Ensure a file/directory was not modified within a block of code.
    """

    def __init__(self, path: Path, msg="") -> None:
        self.path = path
        self.msg = msg
        assert path.exists(), "'unchanging' path doesn't exist originally"

    def __enter__(self):
        __tracebackhide__ = True
        self.original_hashes = _hash_all_files(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        __tracebackhide__ = True
        self._check(self.path, self.msg, self.original_hashes)

    @staticmethod
    def _check(path, msg, original_hashes):
        __tracebackhide__ = True
        assert path.exists(), f"{msg} (deleted! {path})"
        new_hashes = _hash_all_files(path)

        # Convert to sets as pytest output is better
        original_file_list = set(original_hashes.keys())
        new_file_list = set(new_hashes.keys())

        # Are the same set of files in the path?
        assert original_file_list == new_file_list

        # Do they all have the same contents?
        for path_offset, (original_crc32, original_inode) in original_hashes.items():
            new_crc32, new_inode = new_hashes[path_offset]

            assert original_crc32 == new_crc32, f"{msg} (modified: {path_offset})"
            assert original_inode == new_inode, f"{msg} (replaced: {path_offset})"


def _hash_all_files(path: Path) -> dict[Path, tuple[str, int]]:
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
    else:
        files = [path]

    hashes = {}
    for f in files:
        original_crc32 = verify.calculate_file_crc32(f)
        original_inode = f.stat().st_ino
        hashes[f.relative_to(path)] = (original_crc32, original_inode)
    return hashes


def assert_path_eq(p1: Path, p2: Path):
    """Assert two pathlib paths are equal, with reasonable error output."""
    __tracebackhide__ = True
    # Pytest's error messages are far better for strings than Paths.
    # It shows you the difference between them.
    s1, s2 = str(p1), str(p2)
    # And we use extra s1/s2 variables so that pytest doesn't print the
    # expression "str()" as part of its output.
    assert s1 == s2
