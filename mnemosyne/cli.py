"""Command-line interface for the Mnemosyne Protocol."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from mnemosyne import MnemosyneEngine, MnemosyneConfig
from mnemosyne.config import ModelProfile, load_config_from_dict
from mnemosyne.schemas import RetrievalMode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemosyne",
        description="Mnemosyne Protocol — agent memory engine CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a memory block.")
    p_ingest.add_argument("content", help="Memory text content.")
    p_ingest.add_argument("--source", default="cli")
    p_ingest.add_argument("--importance", type=float, default=0.5)

    # retrieve
    p_retrieve = sub.add_parser("retrieve", help="Retrieve memories by query.")
    p_retrieve.add_argument("query", help="Search query text.")
    p_retrieve.add_argument("--top-k", type=int, default=10)
    p_retrieve.add_argument(
        "--mode", choices=["vector", "graph", "hybrid"], default="hybrid"
    )

    # inject
    p_inject = sub.add_parser("inject", help="Build an augmented prompt.")
    p_inject.add_argument("user_input", help="Current user input.")
    p_inject.add_argument("--top-k", type=int, default=None)
    p_inject.add_argument("--profile", default=None)

    # complete
    p_complete = sub.add_parser("complete", help="Inject + complete in one shot.")
    p_complete.add_argument("user_input", help="Current user input.")
    p_complete.add_argument("--profile", default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = MnemosyneConfig()
    config.register_profile(ModelProfile.openai())
    engine = MnemosyneEngine(config)

    try:
        if args.command == "ingest":
            block = engine.ingest(
                args.content, source=args.source,
                importance_score=args.importance,
            )
            print(f"Ingested block {block.id}")
        elif args.command == "retrieve":
            mode = RetrievalMode(args.mode)
            blocks = engine.retrieve(
                args.query, top_k=args.top_k, mode=mode,
            )
            for i, b in enumerate(blocks, 1):
                print(f"[{i}] {b.id} score~{b.importance_score:.2f} :: {b.content[:120]}")
        elif args.command == "inject":
            result = engine.inject(args.user_input, top_k=args.top_k)
            print(result.prompt)
        elif args.command == "complete":
            response = engine.inject_and_complete(
                args.user_input, profile_name=args.profile,
            )
            print(response)
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
