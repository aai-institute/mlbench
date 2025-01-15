import os
from typing import Any


class IOMixin:
    """
    A mixin class providing file I/O operations for benchmark reporters.

    This mixin offers common file handling functionalities like opening, closing, reading and writing.
    This class can be used to create any BenchmarkReporter which has requirement of reading file.
    """

    def open(self, file_path: str) -> None:
        """
        Opens a existing file at a given file_path for read and write operations.

        Parameters
        ----------
        file_path : str
            The path to the target file or directory.

        Raises
        ------
        FileNotFoundError
            If the target file is not present at the file_path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError
        self.file_path: str = file_path
        with open(file_path, "r") as file:
            self.raw_records = file.read()
        self.file = open(file_path, "w")

    def close(self, **kwargs: dict[str, Any]) -> None:
        """Closes the currently open file."""
        if self.file:
            self.file.close()
            self.file = None
            self.file_path = None

    def read_records(self) -> Any:
        return self.raw_records

    def write_records(self, content: str) -> None:
        self.file.write(content)
