"""File-system helper utilities for spec4ml_py workflows."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


def file_hash(path, chunk_size=1024 * 1024):
    """Return the SHA256 hash of a file.

    Parameters
    ----------
    path : str or pathlib.Path
        File path to hash.
    chunk_size : int, optional
        Number of bytes read per chunk. The default is 1 MB.

    Returns
    -------
    str
        SHA256 hash digest.
    """
    path = Path(path)
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def copy_files_from_subfolders(
    source_root,
    destination_folder,
    patterns=("*.csv",),
    conflict_policy="prefix_parent",
    dry_run=False,
):
    """Copy files from nested subfolders into one destination folder.

    Duplicate handling
    ------------------
    - Same filename and identical content: skip duplicate.
    - Same filename but different content: handle according to ``conflict_policy``.

    Parameters
    ----------
    source_root : str or pathlib.Path
        Root folder containing subfolders to search recursively.
    destination_folder : str or pathlib.Path
        Folder where files will be copied.
    patterns : str or iterable of str, optional
        Glob pattern(s) to copy, e.g. ``"*.csv"`` or ``("*.csv", "*.txt")``.
    conflict_policy : {"prefix_parent", "suffix_counter", "error", "overwrite"}, optional
        How to handle files with the same filename but different content.
        ``"prefix_parent"`` renames using the relative parent folder.
        ``"suffix_counter"`` renames using ``__dup1``, ``__dup2``, etc.
        ``"error"`` raises an error.
        ``"overwrite"`` replaces the existing destination file.
    dry_run : bool, optional
        If True, return the planned copy log without copying files or writing the log.

    Returns
    -------
    pandas.DataFrame
        Copy log with source path, destination path, filenames, status, and notes.
    """
    valid_policies = {"prefix_parent", "suffix_counter", "error", "overwrite"}
    if conflict_policy not in valid_policies:
        raise ValueError(
            "conflict_policy must be one of: " + ", ".join(sorted(valid_policies))
        )

    source_root = Path(source_root)
    destination_folder = Path(destination_folder)

    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"source_root is not a directory: {source_root}")

    if isinstance(patterns, (str, bytes)):
        patterns = (patterns,)
    elif not isinstance(patterns, Iterable):
        raise TypeError("patterns must be a string or an iterable of strings")

    if not dry_run:
        destination_folder.mkdir(parents=True, exist_ok=True)

    records = []
    files = []
    for pattern in patterns:
        files.extend(source_root.rglob(pattern))

    files = sorted({path.resolve() for path in files if path.is_file()})
    destination_resolved = destination_folder.resolve()

    for src in files:
        # Avoid recursively copying files that are already inside destination_folder.
        try:
            src.relative_to(destination_resolved)
            continue
        except ValueError:
            pass

        target = destination_folder / src.name
        status = "copied"
        note = ""

        if target.exists():
            src_hash = file_hash(src)
            target_hash = file_hash(target)

            if src_hash == target_hash:
                status = "skipped_identical_duplicate"
                note = "Same filename and identical content already exists."
                records.append(
                    {
                        "source_path": str(src),
                        "destination_path": str(target),
                        "original_filename": src.name,
                        "copied_filename": target.name,
                        "status": status,
                        "note": note,
                    }
                )
                continue

            if conflict_policy == "error":
                raise FileExistsError(
                    "Duplicate filename with different content: "
                    f"{src} conflicts with {target}"
                )

            if conflict_policy == "overwrite":
                status = "overwritten_different_content"
                note = "Same filename existed with different content and was overwritten."

            elif conflict_policy == "prefix_parent":
                try:
                    parent_label = "__".join(src.parent.relative_to(source_root.resolve()).parts)
                except ValueError:
                    parent_label = src.parent.name
                if not parent_label:
                    parent_label = "root"

                target = destination_folder / f"{parent_label}__{src.name}"
                counter = 1

                while target.exists():
                    if file_hash(src) == file_hash(target):
                        status = "skipped_identical_duplicate"
                        note = "Renamed target already exists with identical content."
                        break

                    target = destination_folder / (
                        f"{parent_label}__{src.stem}__dup{counter}{src.suffix}"
                    )
                    counter += 1

                if status != "skipped_identical_duplicate":
                    status = "copied_renamed_conflict"
                    note = (
                        "Same filename existed with different content; "
                        "copied with parent-folder prefix."
                    )

            elif conflict_policy == "suffix_counter":
                counter = 1
                target = destination_folder / f"{src.stem}__dup{counter}{src.suffix}"

                while target.exists():
                    if file_hash(src) == file_hash(target):
                        status = "skipped_identical_duplicate"
                        note = "Renamed target already exists with identical content."
                        break

                    counter += 1
                    target = destination_folder / f"{src.stem}__dup{counter}{src.suffix}"

                if status != "skipped_identical_duplicate":
                    status = "copied_renamed_conflict"
                    note = (
                        "Same filename existed with different content; "
                        "copied with duplicate suffix."
                    )

        if status != "skipped_identical_duplicate" and not dry_run:
            shutil.copy2(src, target)

        records.append(
            {
                "source_path": str(src),
                "destination_path": str(target),
                "original_filename": src.name,
                "copied_filename": target.name,
                "status": status,
                "note": note,
            }
        )

    copy_log = pd.DataFrame(records)

    if not dry_run:
        copy_log.to_csv(destination_folder / "copy_log.csv", index=False)

    return copy_log
