"""File handling helpers for Tabular multipart requests."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from os import PathLike
from pathlib import Path
from typing import BinaryIO

from .models import TabularFile


@contextmanager
def open_tabular_files(
    train_file: TabularFile,
    test_file: TabularFile,
) -> Generator[dict[str, tuple[str, BinaryIO, str]]]:
    """Prepare multipart file tuples, closing only file handles opened here."""
    with ExitStack() as stack:
        files = {
            "train_file": _multipart_file(stack, train_file, "train.csv"),
            "test_file": _multipart_file(stack, test_file, "test.csv"),
        }
        yield files


def _multipart_file(
    stack: ExitStack,
    source: TabularFile,
    default_name: str,
) -> tuple[str, BinaryIO, str]:
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        handle = stack.enter_context(path.open("rb"))
        return path.name, handle, "text/csv"

    if not hasattr(source, "read"):
        raise TypeError("Tabular files must be paths or binary file objects")

    name = getattr(source, "name", default_name)
    filename = Path(str(name)).name if name else default_name
    return filename, source, "text/csv"
