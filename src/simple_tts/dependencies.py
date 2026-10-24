"""Check and install Pandoc and FFmpeg."""

import shutil
import subprocess
from typing import NamedTuple

from rich.console import Console


class Dependency(NamedTuple):
    """A command whose package name matches on Homebrew and APT."""

    name: str
    description: str


DEPENDENCIES: list[Dependency] = [
    Dependency("pandoc", "Document conversion"),
    Dependency("ffmpeg", "Audio encoding"),
]


def is_command_available(command: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(command) is not None


def get_missing_dependencies() -> list[Dependency]:
    """Get missing system dependencies."""
    return [dep for dep in DEPENDENCIES if not is_command_available(dep.name)]


def check_dependencies(console: Console | None = None, warn: bool = True) -> bool:
    """Check required tools and optionally print installation commands."""
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
        packages = " ".join(dep.name for dep in missing)
        console.print(f"  [cyan]brew install {packages}[/cyan]")

        console.print()
        console.print("[bold]To install on Ubuntu/Debian:[/bold]")
        console.print(f"  [cyan]sudo apt-get install {packages}[/cyan]")

        console.print()
        console.print("[dim]Or run: simple-tts install-deps --brew[/dim]")

    return False


def install_dependencies_brew(console: Console | None = None) -> bool:
    """Install missing dependencies using Homebrew."""
    if console is None:
        console = Console()

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
        console.print(f"  Installing {dep.name}...")
        try:
            subprocess.run(
                ["brew", "install", dep.name],
                capture_output=True,
                text=True,
                check=True,
            )
            console.print(f"  [green]✓[/green] Installed {dep.name}")
        except subprocess.CalledProcessError as e:
            console.print(f"  [red]✗[/red] Failed to install {dep.name}: {e.stderr}")
            return False

    console.print("[bold green]✓[/bold green] All dependencies installed!")
    return True


def print_dependency_status(console: Console | None = None) -> None:
    """Print the status of each required tool."""
    if console is None:
        console = Console()

    console.print("[bold]System Dependencies Status:[/bold]")
    console.print()

    for dep in DEPENDENCIES:
        if is_command_available(dep.name):
            console.print(f"  [green]✓[/green] {dep.name} - {dep.description}")
        else:
            console.print(f"  [red]✗[/red] {dep.name} - {dep.description}")
