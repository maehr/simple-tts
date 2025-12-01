"""System dependencies checker for simple-tts.

This module provides utilities for checking and installing system
dependencies required by simple-tts (pdftotext, pandoc, ffmpeg).

Examples:
    >>> from simple_tts.dependencies import is_command_available
    >>> # Check if a command is available
    >>> isinstance(is_command_available("python"), bool)
    True
"""

import shutil
import subprocess
from typing import NamedTuple

from rich.console import Console


class Dependency(NamedTuple):
    """System dependency information.

    Attributes:
        name: Command name to check.
        package: Package name for installation.
        brew_package: Homebrew package name (may differ from package).
        apt_package: APT package name for Debian/Ubuntu.
        description: Human-readable description.
    """

    name: str
    package: str
    brew_package: str
    apt_package: str
    description: str


# Required system dependencies
DEPENDENCIES: list[Dependency] = [
    Dependency(
        name="pdftotext",
        package="poppler",
        brew_package="poppler",
        apt_package="poppler-utils",
        description="PDF text extraction (from Poppler)",
    ),
    Dependency(
        name="pandoc",
        package="pandoc",
        brew_package="pandoc",
        apt_package="pandoc",
        description="Universal document converter",
    ),
    Dependency(
        name="ffmpeg",
        package="ffmpeg",
        brew_package="ffmpeg",
        apt_package="ffmpeg",
        description="Audio encoding and processing",
    ),
]


def is_command_available(command: str) -> bool:
    """Check if a command is available in PATH.

    Args:
        command: Name of the command to check.

    Returns:
        True if the command is found, False otherwise.

    Examples:
        >>> is_command_available("python")
        True
        >>> is_command_available("nonexistent_command_12345")
        False
    """
    return shutil.which(command) is not None


def get_missing_dependencies() -> list[Dependency]:
    """Get list of missing system dependencies.

    Returns:
        List of Dependency objects for commands not found in PATH.

    Examples:
        >>> deps = get_missing_dependencies()
        >>> isinstance(deps, list)
        True
    """
    return [dep for dep in DEPENDENCIES if not is_command_available(dep.name)]


def check_dependencies(console: Console | None = None, warn: bool = True) -> bool:
    """Check if all required system dependencies are installed.

    Args:
        console: Rich console for output. If None, uses default.
        warn: Whether to print warnings for missing dependencies.

    Returns:
        True if all dependencies are installed, False otherwise.

    Examples:
        >>> result = check_dependencies(warn=False)
        >>> isinstance(result, bool)
        True
    """
    if console is None:
        console = Console()

    missing = get_missing_dependencies()

    if not missing:
        return True

    if warn:
        console.print("[bold yellow]⚠️  Missing system dependencies:[/bold yellow]")
        for dep in missing:
            console.print(f"  • [red]{dep.name}[/red] - {dep.description}")

        console.print()
        console.print("[bold]To install on macOS (Homebrew):[/bold]")
        brew_packages = " ".join(dep.brew_package for dep in missing)
        console.print(f"  [cyan]brew install {brew_packages}[/cyan]")

        console.print()
        console.print("[bold]To install on Ubuntu/Debian:[/bold]")
        apt_packages = " ".join(dep.apt_package for dep in missing)
        console.print(f"  [cyan]sudo apt-get install {apt_packages}[/cyan]")

        console.print()
        console.print("[dim]Or run: simple-tts install-deps --brew[/dim]")

    return False


def install_dependencies_brew(console: Console | None = None) -> bool:
    """Install missing dependencies using Homebrew.

    Args:
        console: Rich console for output. If None, uses default.

    Returns:
        True if installation succeeded, False otherwise.

    Raises:
        RuntimeError: If Homebrew is not installed.
    """
    if console is None:
        console = Console()

    # Check if brew is available
    if not is_command_available("brew"):
        console.print("[bold red]Error:[/bold red] Homebrew is not installed.")
        console.print("Install Homebrew from: [cyan]https://brew.sh[/cyan]")
        return False

    missing = get_missing_dependencies()

    if not missing:
        console.print("[bold green]✓[/bold green] All dependencies are installed!")
        return True

    console.print("[bold blue]Installing dependencies with Homebrew...[/bold blue]")

    for dep in missing:
        console.print(f"  Installing {dep.brew_package}...")
        try:
            subprocess.run(
                ["brew", "install", dep.brew_package],
                capture_output=True,
                text=True,
                check=True,
            )
            console.print(f"  [green]✓[/green] Installed {dep.brew_package}")
        except subprocess.CalledProcessError as e:
            console.print(
                f"  [red]✗[/red] Failed to install {dep.brew_package}: {e.stderr}"
            )
            return False

    console.print("[bold green]✓[/bold green] All dependencies installed!")
    return True


def get_dependency_status() -> dict[str, bool]:
    """Get installation status of all dependencies.

    Returns:
        Dictionary mapping dependency names to installation status.

    Examples:
        >>> status = get_dependency_status()
        >>> isinstance(status, dict)
        True
        >>> all(isinstance(v, bool) for v in status.values())
        True
    """
    return {dep.name: is_command_available(dep.name) for dep in DEPENDENCIES}


def print_dependency_status(console: Console | None = None) -> None:
    """Print a formatted status of all dependencies.

    Args:
        console: Rich console for output. If None, uses default.
    """
    if console is None:
        console = Console()

    console.print("[bold]System Dependencies Status:[/bold]")
    console.print()

    for dep in DEPENDENCIES:
        if is_command_available(dep.name):
            console.print(f"  [green]✓[/green] {dep.name} - {dep.description}")
        else:
            console.print(f"  [red]✗[/red] {dep.name} - {dep.description}")
