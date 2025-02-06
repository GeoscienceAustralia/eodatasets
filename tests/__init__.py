import atexit
import os
import pathlib
import shutil
import sys
import tempfile
from pathlib import Path


def assert_same(o1, o2, prefix=""):
    """
    Assert the two are equal.

    Compares property values one-by-one recursively to print friendly error messages.

    (ie. the exact property that differs)

    :type o1: object
    :type o2: object
    :raises: AssertionError
    """
    __tracebackhide__ = True

    def _compare(k, val1, val2):
        assert_same(val1, val2, prefix=prefix + "." + str(k))

    if isinstance(o1, list) and isinstance(o2, list):
        assert len(o1) == len(o2), "Differing lengths: %s" % prefix

        for i, val in enumerate(o1):
            _compare(i, val, o2[i])
    elif isinstance(o1, dict) and isinstance(o2, dict):
        for k, val in o1.items():
            assert k in o2, f"{prefix}[{k!r}] is missing.\n\t{o1!r}\n\t{o2!r}"
        for k, val in o2.items():
            assert k in o1, f"{prefix}[{k!r}] is missing.\n\t{o2!r}\n\t{o1!r}"
            _compare(k, val, o1[k])
    elif o1 != o2:
        sys.stderr.write("%r\n" % o1)
        sys.stderr.write("%r\n" % o2)
        raise AssertionError(f"Mismatch for property {prefix!r}:  {o1!r} != {o2!r}")


def assert_file_structure(folder, expected_structure, root=""):
    """
    Assert that the contents of a folder (filenames and subfolder names recursively)
    match the given nested dictionary structure.

    :type folder: pathlib.Path
    :type expected_structure: dict[str,str|dict]
    """
    __tracebackhide__ = True
    required_filenames = {
        name for name, option in expected_structure.items() if option != "optional"
    }
    optional_filenames = {
        name for name, option in expected_structure.items() if option == "optional"
    }
    assert folder.exists(), (
        f"Expected base folder doesn't even exist! {folder.as_posix()!r}"
    )

    actual_filenames = {f.name for f in folder.iterdir()}

    if required_filenames != (actual_filenames - optional_filenames):
        missing_files = required_filenames - actual_filenames
        missing_text = "Missing: %r" % sorted(missing_files)
        extra_files = actual_filenames - required_filenames
        added_text = "Extra  : %r" % sorted(extra_files)
        raise AssertionError(
            f"Folder mismatch of {root!r}\n\t{missing_text}\n\t{added_text}"
        )

    for k, v in expected_structure.items():
        id_ = f"{root}/{k}" if root else k

        is_optional = v == "optional"

        f = folder.joinpath(k)

        if not f.exists():
            if is_optional:
                continue

            raise AssertionError(f"{id_} is missing")
        elif isinstance(v, dict):
            assert f.is_dir(), f"{id_} is not a dir"
            assert_file_structure(f, v, id_)
        elif isinstance(v, str):
            assert f.is_file(), f"{id_} is not a file"
        else:
            raise AssertionError(
                "Only strings and dicts expected when defining a folder structure."
            )


def write_files(file_dict):
    """
    Convenience method for writing a bunch of files to a temporary directory.

    Dict format is "filename": "text content"

    If content is another dict, it is created recursively in the same manner.

    writeFiles({'test.txt': 'contents of text file'})

    :type file_dict: dict
    :rtype: pathlib.Path
    :return: Created temporary directory path
    """
    containing_dir = tempfile.mkdtemp(suffix="neotestrun")
    _write_files_to_dir(containing_dir, file_dict)

    def remove_if_exists(path):
        if os.path.exists(path):
            shutil.rmtree(path)

    atexit.register(remove_if_exists, containing_dir)
    return pathlib.Path(containing_dir)


def _write_files_to_dir(directory_path, file_dict):
    """
    Convenience method for writing a bunch of files to a given directory.

    :type directory_path: str
    :type file_dict: dict
    """
    for filename, contents in file_dict.items():
        path = os.path.join(directory_path, filename)
        if isinstance(contents, dict):
            os.mkdir(path)
            _write_files_to_dir(path, contents)
        else:
            with open(path, "w") as f:
                if isinstance(contents, list):
                    f.writelines(contents)
                elif isinstance(contents, str):
                    f.write(contents)
                else:
                    raise Exception("Unexpected file contents: %s" % type(contents))


def temp_dir():
    """
    Create and return a temporary directory that will be deleted automatically on exit.

    :rtype: pathlib.Path
    """
    return write_files({})


def file_of_size(path, size_mb):
    """
    Create a blank file of the given size.
    """
    with open(path, "wb") as f:
        f.seek(size_mb * 1024 * 1024 - 1)
        f.write(b"\0")


def as_file_list(path):
    """
    Build a flat list of filenames relative to the given folder
    (similar to the contents of package.sha1 files)
    """
    output = []
    for directory, _, files in os.walk(str(path)):
        output.extend(
            str(Path(directory).relative_to(path).joinpath(file_)) for file_ in files
        )
    return output
