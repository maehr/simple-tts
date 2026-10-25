"""Confine MCP inputs and atomic output publication to one workspace."""

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Workspace:
    """Resolve paths and use directory handles for file operations."""

    def __init__(self, path: Path) -> None:
        """Require an existing workspace directory."""
        self.path = path.resolve(strict=True)
        if not self.path.is_dir():
            raise ValueError("The workspace must be an existing directory.")

    def resolve(self, path: Path) -> Path:
        """Reject resolved paths outside the workspace, including symlinks."""
        candidate = path if path.is_absolute() else self.path / path
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.path):
            raise ValueError(f"Path escapes the workspace: {path}")
        return resolved

    @contextmanager
    def directory(self, path: Path, *, create: bool = False) -> Iterator[int]:
        """Open each directory without following replacement symlinks."""
        relative = path.relative_to(self.path)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        try:
            for part in relative.parts:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor
        finally:
            os.close(descriptor)

    def default_output(self, job_id: str, format_type: str) -> Path:
        """Create the default audio directory within the workspace."""
        directory = self.path / ".simple-tts" / "audio"
        with self.directory(directory, create=True):
            pass
        return directory / f"{job_id}.{format_type}"

    def check_output(self, path: Path) -> Path:
        """Require an existing parent and a free output path."""
        # Check the original leaf too: a dangling symlink is an existing output.
        original = path if path.is_absolute() else self.path / path
        resolved = self.resolve(original)
        if original.is_symlink() or resolved.exists():
            raise ValueError(f"Output already exists: {original}")
        with self.directory(resolved.parent):
            pass
        return resolved

    def copy_input(self, source: Path, destination: Path) -> None:
        """Snapshot a source without following symlinks changed after validation."""
        if self.resolve(source) != source:
            raise ValueError("The input path changed after validation.")
        with self.directory(source.parent) as parent:
            descriptor = os.open(
                source.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent
            )
            with (
                os.fdopen(descriptor, "rb") as stream,
                destination.open("wb") as output,
            ):
                shutil.copyfileobj(stream, output)

    def publish(self, staged: Path, output: Path) -> None:
        """Atomically publish a complete file without replacing another file."""
        if self.check_output(output) != output:
            raise ValueError("The output path changed after validation.")
        with (
            self.directory(output.parent) as parent,
            self.directory(staged.parent) as temporary,
        ):
            os.link(
                staged.name,
                output.name,
                src_dir_fd=temporary,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )


class ScratchDirectory:
    """Keep a directory handle so cleanup survives parent path changes."""

    def __init__(self, workspace: Workspace, parent: Path) -> None:
        """Create scratch storage through a validated parent handle."""
        with workspace.directory(parent) as descriptor:
            self._parent = os.dup(descriptor)
        self._leaf = f".simple-tts-{uuid4().hex}"
        self.name = str(parent / self._leaf)
        try:
            os.mkdir(self._leaf, mode=0o700, dir_fd=self._parent)
        except BaseException:
            os.close(self._parent)
            raise

    def cleanup(self) -> None:
        """Remove temporary files even if the parent was renamed."""
        try:
            shutil.rmtree(self._leaf, dir_fd=self._parent)
        finally:
            os.close(self._parent)
