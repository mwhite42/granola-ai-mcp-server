#!/usr/bin/env python3
"""Run the Granola MCP Server."""

import os
import sys

if os.getenv("GRANOLA_DEBUG"):
    print(f"Granola MCP Server starting with PID: {os.getpid()}", file=sys.stderr)
    print(f"Python path: {sys.executable}", file=sys.stderr)
    print(f"Working directory: {os.getcwd()}", file=sys.stderr)

from granola_mcp_server.server import main

if __name__ == "__main__":
    if os.getenv("GRANOLA_DEBUG"):
        print("About to call main()", file=sys.stderr)
    main()
    if os.getenv("GRANOLA_DEBUG"):
        print("main() completed", file=sys.stderr)
