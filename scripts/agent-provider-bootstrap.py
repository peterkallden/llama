#!/usr/bin/env python3
"""Create one host-owned MCP or OpenAPI provider fragment."""

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request


def fetch_openapi(base_url, spec_output):
    base = base_url.rstrip("/")
    candidates = ("/openapi.json", "/swagger.json", "/api-docs")
    opener = urllib.request.build_opener(
        urllib.request.HTTPRedirectHandler())
    for suffix in candidates:
        url = base + suffix
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with opener.open(request, timeout=5) as response:
                if response.status < 200 or response.status >= 300:
                    continue
                payload = response.read()
                document = json.loads(payload)
                if not isinstance(document, dict):
                    continue
                pathlib.Path(spec_output).write_bytes(payload)
                return url
        except (OSError, ValueError, urllib.error.URLError):
            continue
    raise SystemExit("could not discover an OpenAPI document; use --spec")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("openapi", "mcp"), required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--spec")
    parser.add_argument("--spec-output")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--transport", choices=("stdio", "streamable_http"), default="stdio")
    parser.add_argument("--url")
    parser.add_argument("--server-name")
    parser.add_argument("--command", nargs="+")
    parser.add_argument("--auth-type", choices=("none", "bearer"), default="none")
    parser.add_argument("--token-env")
    parser.add_argument("--allowed-tool", action="append", default=[])
    args = parser.parse_args()

    if args.auth_type == "bearer" and not args.token_env:
        parser.error("--auth-type=bearer requires --token-env")

    provider = {"type": args.type, "id": args.id, "enabled": True}
    if args.prefix:
        provider["prefix"] = args.prefix
    provider["auth"] = {"type": args.auth_type}
    if args.auth_type == "bearer":
        provider["auth"]["token_env"] = args.token_env

    if args.type == "openapi":
        if not args.base_url:
            parser.error("OpenAPI providers require --base-url")
        provider["base_url"] = args.base_url
        spec = args.spec
        if not spec:
            spec = args.spec_output or (args.id + "-openapi.json")
            discovered = fetch_openapi(args.base_url, spec)
            print("discovered OpenAPI document: " + discovered, file=sys.stderr)
        provider["spec_path"] = spec
    else:
        if args.transport == "stdio" and not args.command:
            parser.error("stdio MCP providers require --command")
        if args.transport != "stdio" and not args.url:
            parser.error("HTTP MCP providers require --url")
        provider["transport"] = args.transport
        if args.url:
            provider["url"] = args.url
        if args.command:
            provider["command"] = args.command
        if args.server_name:
            provider["server_name"] = args.server_name
        if args.allowed_tool:
            provider["allowed_tools"] = args.allowed_tool

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
