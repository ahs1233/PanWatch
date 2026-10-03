"""Manage the isolated Graphify knowledge layer for PanWatch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".graphify-venv"
GRAPH_PATH = ROOT / "graphify-out" / "graph.json"
REQUIREMENTS = ROOT / "requirements-graphify.txt"


def _python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _require_install() -> Path:
    python = _python()
    if not python.exists():
        raise SystemExit(
            "Graphify is not installed. Run: python scripts/graphify_knowledge.py install"
        )
    return python


def _run_graphify(*args: str) -> None:
    subprocess.run(
        [str(_require_install()), "-m", "graphify", *args],
        cwd=ROOT,
        check=True,
    )


def install() -> None:
    if not _python().exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)
    subprocess.run(
        [str(_python()), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(_python()), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        cwd=ROOT,
        check=True,
    )


def build(*, semantic: bool, backend: str | None, force: bool) -> None:
    args = ["extract", "."]
    if not semantic:
        args.append("--code-only")
    elif backend:
        args.extend(["--backend", backend])
    if force:
        args.append("--force")
    _run_graphify(*args)


def update(*, force: bool) -> None:
    args = ["update", "."]
    if force:
        args.append("--force")
    _run_graphify(*args)


def serve(host: str, port: int, api_key: str) -> None:
    if not GRAPH_PATH.exists():
        raise SystemExit(
            "graphify-out/graph.json does not exist. Run the build command first."
        )
    args = [
        str(_require_install()),
        "-m",
        "graphify.serve",
        str(GRAPH_PATH),
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--json-response",
    ]
    if api_key:
        args.extend(["--api-key", api_key])
    subprocess.run(args, cwd=ROOT, check=True)


def query(question: str) -> None:
    if not GRAPH_PATH.exists():
        raise SystemExit(
            "graphify-out/graph.json does not exist. Run the build command first."
        )
    _run_graphify("query", question)


def hook() -> None:
    _run_graphify("hook", "install")


def main() -> None:
    parser = argparse.ArgumentParser(description="PanWatch Graphify knowledge layer")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("install", help="create isolated Graphify environment")

    build_parser = sub.add_parser("build", help="build graphify-out/graph.json")
    build_parser.add_argument(
        "--semantic",
        action="store_true",
        help="include docs/media semantic extraction",
    )
    build_parser.add_argument(
        "--backend",
        choices=[
            "gemini",
            "kimi",
            "claude",
            "openai",
            "deepseek",
            "ollama",
            "bedrock",
            "claude-cli",
            "azure",
        ],
    )
    build_parser.add_argument("--force", action="store_true")

    update_parser = sub.add_parser("update", help="incrementally update the graph")
    update_parser.add_argument("--force", action="store_true")

    serve_parser = sub.add_parser("serve", help="serve graph over Streamable HTTP MCP")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8766)
    serve_parser.add_argument(
        "--api-key",
        default=os.environ.get("GRAPHIFY_API_KEY", ""),
    )

    query_parser = sub.add_parser("query", help="query the local graph")
    query_parser.add_argument("question")

    sub.add_parser("hook", help="install Graphify post-commit update hook")

    args = parser.parse_args()
    if args.command == "install":
        install()
    elif args.command == "build":
        build(semantic=args.semantic, backend=args.backend, force=args.force)
    elif args.command == "update":
        update(force=args.force)
    elif args.command == "serve":
        serve(args.host, args.port, args.api_key)
    elif args.command == "query":
        query(args.question)
    elif args.command == "hook":
        hook()


if __name__ == "__main__":
    main()
