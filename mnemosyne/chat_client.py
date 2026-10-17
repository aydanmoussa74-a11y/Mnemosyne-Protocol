"""Mnemosyne Protocol — real-time CLI chat loop.

An interactive terminal chat interface that connects directly to a local
Mnemosyne engine and a configured model adapter (Ollama, OpenAI, ...).

Every user turn:
    1. Runs hybrid retrieval against the memory store.
    2. Injects micro-targeted historical anchors via the token-budget middleware.
    3. Forwards the augmented prompt to the model adapter.
    4. Prints the live (streaming) response.
    5. Ingests the user turn + assistant response back into the store.

Designed to run standalone against a local engine (no HTTP server required),
or against a running FastAPI microservice (``--remote``).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.table import Table

from mnemosyne import MnemosyneEngine, MnemosyneConfig
from mnemosyne.adapters import build_adapter
from mnemosyne.config import ModelProfile
from mnemosyne.exceptions import MnemosyneError
from mnemosyne.injection import InjectionResult
from mnemosyne.schemas import RetrievalMode

console = Console()


# --------------------------------------------------------------------------- #
# Chat client
# --------------------------------------------------------------------------- #
class MnemosyneChatClient:
    """Interactive REPL for chatting with a memory-augmented model."""

    def __init__(
        self,
        engine: MnemosyneEngine,
        profile_name: Optional[str] = None,
        *,
        top_k: int = 8,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        auto_ingest: bool = True,
    ) -> None:
        self.engine = engine
        self.profile_name = profile_name or engine.config.default_profile
        self.top_k = top_k
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.auto_ingest = auto_ingest
        self.adapter = self._resolve_adapter()
        self._turn_count = 0

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _resolve_adapter(self):
        profile = self.engine.config.get_profile(self.profile_name)
        return build_adapter(profile)

    def switch_profile(self, name: str) -> None:
        self.profile_name = name
        self.adapter = self._resolve_adapter()
        console.print(f"[dim]Switched to model profile: {name}[/dim]")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Enter the interactive REPL."""
        self._print_banner()

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break

            if not user_input:
                continue

            # Commands.
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            self._process_turn(user_input)

    # ------------------------------------------------------------------ #
    # Turn processing
    # ------------------------------------------------------------------ #
    def _process_turn(self, user_input: str) -> None:
        self._turn_count += 1
        turn_id = self._turn_count

        # 1. Ingest the user turn into memory (short-term, high recency).
        if self.auto_ingest:
            self.engine.ingest(
                content=f"[user] {user_input}",
                source="chat",
                importance_score=0.6,
                tags=["chat", f"turn:{turn_id}", "user"],
            )

        # 2. Retrieve + inject context.
        try:
            injection: InjectionResult = self.engine.inject(
                user_input=user_input,
                model_name=self.engine.config.get_profile(
                    self.profile_name
                ).model,
                top_k=self.top_k,
                mode=self.mode,
            )
        except MnemosyneError as exc:
            console.print(f"[red]Injection error:[/red] {exc.message}")
            return

        # 3. Complete via model adapter (streaming).
        profile = self.engine.config.get_profile(self.profile_name)

        console.print("\n[bold cyan]Assistant[/bold cyan]")

        try:
            response_text = self.adapter.complete(
                injection,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except MnemosyneError as exc:
            console.print(f"[red]Model error:[/red] {exc.message}")
            return
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected model error:[/red] {exc}")
            return

        # Render the response as markdown.
        console.print(Markdown(response_text))

        # 4. Ingest the assistant turn back into memory.
        if self.auto_ingest and response_text.strip():
            self.engine.ingest(
                content=f"[assistant] {response_text}",
                source="chat",
                importance_score=0.55,
                tags=["chat", f"turn:{turn_id}", "assistant"],
            )

        # 5. Show a compact context summary.
        self._show_context_summary(injection)

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    def _handle_command(self, raw: str) -> None:
        parts = raw[1:].strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("q", "quit", "exit"):
            console.print("[bold yellow]Goodbye![/bold yellow]")
            sys.exit(0)

        elif cmd == "profile":
            if not arg:
                self._show_profiles()
            else:
                self.switch_profile(arg)

        elif cmd == "stats":
            self._show_stats()

        elif cmd == "history":
            self._show_history(limit=int(arg) if arg.isdigit() else 10)

        elif cmd == "clear":
            console.clear()

        elif cmd in ("h", "help"):
            self._print_help()

        else:
            console.print(f"[red]Unknown command:[/red] /{cmd}. Type /help.")

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #
    def _print_banner(self) -> None:
        profile = self.engine.config.get_profile(self.profile_name)
        banner = Text()
        banner.append("Mnemosyne Protocol", style="bold magenta")
        banner.append("  |  memory-augmented chat", style="dim")
        console.print(Panel(banner, expand=False))
        console.print(
            f"[dim]Profile:[/dim] {profile.name} ({profile.provider.value}) "
            f"| model={profile.model} | mode={self.mode.value} | top_k={self.top_k}"
        )
        console.print(f"[dim]/profile <name> | /stats | /history [n] | /clear | /quit[/dim]")

    def _show_context_summary(self, injection: InjectionResult) -> None:
        budget = injection.budget
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("label", style="dim", width=22)
        table.add_column("value")
        table.add_row("Injected blocks", str(len(injection.injected_blocks)))
        table.add_row("Injected tokens", str(injection.total_injected_tokens))
        table.add_row("Budget usable", str(budget.usable_for_retrieval))
        table.add_row("Mode", self.mode.value)
        console.print(
            Panel(table, title="[dim]context[/dim]", expand=False, style="dim")
        )

    def _show_profiles(self) -> None:
        table = Table(title="Model Profiles")
        table.add_column("Name")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Default", style="green")
        for name, p in self.engine.config.profiles.items():
            table.add_row(
                name,
                p.provider.value,
                p.model,
                "✓" if name == self.engine.config.default_profile else "",
            )
        console.print(table)

    def _show_stats(self) -> None:
        block_count = self.engine.store.count_blocks()
        jobs = self.engine.list_consolidation_jobs()
        table = Table(title="Engine Stats")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Blocks stored", str(block_count))
        table.add_row("Consolidation jobs", str(len(jobs)))
        table.add_row("Default profile", self.engine.config.default_profile or "—")
        console.print(table)

    def _show_history(self, limit: int = 10) -> None:
        blocks = self.engine.store.list_blocks(limit=limit, offset=0)
        chat_blocks = [
            b for b in blocks if b.source == "chat"
        ][:limit]
        if not chat_blocks:
            console.print("[dim]No chat history yet.[/dim]")
            return
        table = Table(title=f"Recent Chat History (last {len(chat_blocks)})")
        table.add_column("Turn", style="cyan", width=6)
        table.add_column("Role", width=10)
        table.add_column("Content")
        for b in reversed(chat_blocks):
            role = "user" if b.content.startswith("[user]") else "assistant"
            content = b.content.split("] ", 1)[1] if "] " in b.content else b.content
            tag = "[user]" if role == "user" else "[assistant]"
            style = "green" if role == "user" else "cyan"
            table.add_row(
                str(b.id)[:8],
                Text(tag, style=style),
                Text(content[:100], style=style),
            )
        console.print(table)

    def _print_help(self) -> None:
        help_text = """
[bold]Mnemosyne Chat Client — commands[/bold]

  /profile <name>   Switch model profile (see /profiles)
  /stats            Show engine statistics
  /history [n]      Show last n chat turns
  /clear            Clear the screen
  /quit             Exit the chat

Type any other message to chat. Each turn is automatically stored in memory
and retrieved contextually on future turns.
        """
        console.print(Panel(help_text.strip(), title="Help", expand=False))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemosyne-chat",
        description="Mnemosyne Protocol — real-time CLI chat loop.",
    )
    parser.add_argument(
        "--storage-path",
        default="./mnemosyne_data",
        help="Path to the Mnemosyne data directory.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Model profile name to use (defaults to engine default).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of memory anchors to retrieve per turn.",
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "graph", "hybrid"],
        default="hybrid",
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max completion tokens.",
    )
    parser.add_argument(
        "--no-auto-ingest",
        action="store_true",
        help="Disable automatic ingestion of chat turns.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = MnemosyneConfig(storage_path=args.storage_path)
    # Register a sensible default if none exist.
    if not config.profiles:
        config.register_profile(ModelProfile.openai(name="default"))

    engine = MnemosyneEngine(config)

    try:
        client = MnemosyneChatClient(
            engine=engine,
            profile_name=args.profile,
            top_k=args.top_k,
            mode=RetrievalMode(args.mode),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            auto_ingest=not args.no_auto_ingest,
        )
        client.run()
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
