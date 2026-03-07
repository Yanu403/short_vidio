"""Rich-based CLI logger helpers."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule

console = Console()


def header(title: str) -> None:
    """Render pipeline header."""
    console.print(Rule(style="cyan"))
    console.print(f"[bold]🎬 {title}[/bold]")
    console.print(Rule(style="cyan"))


def stage(name: str) -> None:
    """Render stage label."""
    console.print(f"\n[bold cyan]▶ {name}[/bold cyan]")


def success(msg: str) -> None:
    """Render success message."""
    console.print(f"[bold green]✓ {msg}[/bold green]")


def error(msg: str) -> None:
    """Render error message."""
    console.print(Panel.fit(f"[bold red]✗ {msg}[/bold red]", border_style="red"))


def info(msg: str) -> None:
    """Render neutral information."""
    console.print(f"[white]{msg}[/white]")
