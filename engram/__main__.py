"""
engram.__main__ -- CLI entry point.

Usage:
    engram init [--data-dir DIR]
    engram serve [--data-dir DIR] [--config PATH] [--transport stdio|sse]
    engram stats [--data-dir DIR]
    engram search QUERY [--person NAME] [--limit N]
    engram reindex [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="engram",
        description="Engram -- four-layer memory system for persistent AI identity",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    sub = parser.add_subparsers(dest="command")

    # -- init --------------------------------------------------------------
    init_p = sub.add_parser("init", help="Initialize a new Engram data directory")
    init_p.add_argument(
        "--data-dir",
        default="./engram_data",
        help="Data directory to create (default: ./engram_data)",
    )

    # -- serve -------------------------------------------------------------
    serve_p = sub.add_parser("serve", help="Start the MCP server")
    serve_p.add_argument("--data-dir", default=None, help="Path to data directory")
    serve_p.add_argument("--config", default=None, help="Path to engram.yaml config")
    serve_p.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="MCP transport (default: stdio)",
    )

    # -- stats -------------------------------------------------------------
    stats_p = sub.add_parser("stats", help="Show memory system statistics")
    stats_p.add_argument("--data-dir", default="./engram_data", help="Data directory")

    # -- search ------------------------------------------------------------
    search_p = sub.add_parser("search", help="Search memory")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--person", default=None, help="Filter by person")
    search_p.add_argument("--limit", type=int, default=20, help="Max results")
    search_p.add_argument("--data-dir", default="./engram_data", help="Data directory")

    # -- reindex -----------------------------------------------------------
    reindex_p = sub.add_parser("reindex", help="Rebuild semantic search indexes")
    reindex_p.add_argument("--data-dir", default="./engram_data", help="Data directory")

    args = parser.parse_args(argv)

    # -- Logging -----------------------------------------------------------
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command is None:
        parser.print_help()
        return

    # -- Dispatch ----------------------------------------------------------
    if args.command == "init":
        _cmd_init(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "stats":
        _cmd_stats(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "reindex":
        _cmd_reindex(args)
    else:
        parser.print_help()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    """Create a fresh Engram data directory with template files."""
    from engram.core.config import Config

    data_dir = Path(args.data_dir).resolve()
    config = Config.from_data_dir(data_dir)
    config.ensure_directories()

    # Create template SOUL.md if it doesn't exist
    soul_path = config.soul_path
    if not soul_path.exists():
        soul_path.write_text(
            "# Identity\n"
            "\n"
            "Who are you? Write your identity document here.\n"
            "\n"
            "This is your SOUL.md -- the core of who you are.\n"
            "It gets loaded into every conversation as foundational context.\n"
            "\n"
            "## Core Values\n"
            "\n"
            "## Personality\n"
            "\n"
            "## Boundaries\n",
            encoding="utf-8",
        )

    # Create template identities.yaml if it doesn't exist
    identities_path = config.identities_path
    if not identities_path.exists():
        identities_path.write_text(
            "# Identity resolution: aliases -> canonical names\n"
            "people:\n"
            "  # example:\n"
            "  #   alice:\n"
            "  #     aliases: [alice_dev, Alice, alice123]\n"
            '  #     discord_id: "123456789"\n'
            "  #     trust_tier: friend\n",
            encoding="utf-8",
        )

    # Create template engram.yaml config
    config_path = data_dir / "engram.yaml"
    if not config_path.exists():
        config_path.write_text(
            "# Engram configuration\n"
            "engram:\n"
            f"  data_dir: {data_dir}\n"
            "  signal_mode: hybrid     # hybrid | regex | llm\n"
            "  extract_mode: off       # llm | off\n"
            "  llm_provider: ollama    # ollama | openai | anthropic\n"
            "  llm_model: llama3.2\n"
            "  llm_base_url: http://localhost:11434\n"
            "  token_budget: 6000\n"
            "  decay_half_life_hours: 168  # 1 week\n"
            "  max_traces: 50000\n",
            encoding="utf-8",
        )

    print(f"Initialized Engram at: {data_dir}")
    print(f"  SOUL.md:         {soul_path}")
    print(f"  identities.yaml: {identities_path}")
    print(f"  engram.yaml:     {config_path}")
    print()
    print("Next steps:")
    print("  1. Edit SOUL.md with your identity")
    print("  2. Add people to identities.yaml")
    print("  3. Run: engram serve --data-dir", str(data_dir))


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    from engram.server import run_server

    run_server(
        data_dir=args.data_dir,
        config_path=args.config,
        transport=args.transport,
    )


def _cmd_stats(args: argparse.Namespace) -> None:
    """Print memory system statistics."""
    from engram.system import MemorySystem

    with MemorySystem(data_dir=args.data_dir) as mem:
        stats = mem.get_stats()
        print(json.dumps(stats.to_dict(), indent=2))


def _cmd_search(args: argparse.Namespace) -> None:
    """Search memory and print results."""
    from engram.system import MemorySystem

    with MemorySystem(data_dir=args.data_dir) as mem:
        results = mem.search(
            query=args.query,
            person=args.person,
            limit=args.limit,
        )
        if results:
            print(json.dumps(results, indent=2, default=str))
        else:
            print("No results found.")


def _cmd_reindex(args: argparse.Namespace) -> None:
    """Rebuild semantic search indexes."""
    from engram.system import MemorySystem

    with MemorySystem(data_dir=args.data_dir) as mem:
        counts = mem.reindex()
        print(json.dumps({"reindexed": counts}, indent=2))


if __name__ == "__main__":
    main()
