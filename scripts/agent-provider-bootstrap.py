#!/usr/bin/env python3
"""Create one host-owned MCP or OpenAPI provider fragment."""

import argparse
import copy
import json
import pathlib
import sys
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, msg, headers, newurl):
        return None


def fetch_openapi(base_url, spec_output):
    base = base_url.rstrip("/")
    candidates = ("/openapi.json", "/swagger.json", "/api-docs")
    return fetch_openapi_url((base + suffix for suffix in candidates), spec_output)


def fetch_openapi_url(urls, spec_output):
    opener = urllib.request.build_opener(
        NoRedirect())
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "llama-agent-provider-bootstrap/1.0",
            })
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
    raise SystemExit("could not fetch an OpenAPI document; use --spec")


def make_public_read_operations_anonymous(spec_output, allowed_operation_ids=None):
    """Override security only for safe operations in an operator-approved public API.

    Some public APIs describe an optional API key as a global OpenAPI security
    requirement.  The native provider correctly treats that declaration as an
    authentication contract, so a host that deliberately uses anonymous public
    reads needs an explicit, auditable override in its generated copy.
    """
    path = pathlib.Path(spec_output)
    document = json.loads(path.read_text(encoding="utf-8"))
    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise SystemExit("OpenAPI document has no paths object")

    components = document.get("components", {})
    named_parameters = components.get("parameters", {}) if isinstance(components, dict) else {}

    def parameter_name(parameter):
        if not isinstance(parameter, dict):
            return ""
        if isinstance(parameter.get("$ref"), str):
            prefix = "#/components/parameters/"
            reference = parameter["$ref"]
            if reference.startswith(prefix):
                parameter = named_parameters.get(reference[len(prefix):], parameter)
        return parameter.get("name", "") if isinstance(parameter, dict) else ""

    changed = 0
    for path_item in paths.values():
        if not isinstance(path_item, dict):
            continue
        for method in ("get", "head", "options"):
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            if allowed_operation_ids and operation.get("operationId") not in allowed_operation_ids:
                continue
            operation["security"] = []
            # A public-read override must also remove the corresponding
            # required API-key query parameter.  Otherwise the generated
            # schema still asks the model/runtime for a credential even though
            # operation security was deliberately made anonymous.
            parameters = operation.get("parameters")
            if isinstance(parameters, list):
                operation["parameters"] = [
                    parameter for parameter in parameters
                    if parameter_name(parameter) not in {"api_key", "api-key", "apikey"}
                ]
            changed += 1
    if changed == 0:
        raise SystemExit("OpenAPI document has no selected safe operations to mark public")
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def parse_operation_values(values, option_name):
    parsed = {}
    for value in values:
        operation_id, separator, setting = value.partition("=")
        if not separator or not operation_id.strip() or not setting.strip():
            raise SystemExit(f"{option_name} must use OPERATION_ID=VALUE")
        parsed[operation_id.strip()] = setting.strip()
    return parsed


def validate_path_parameter_prefixes(spec_output, prefixes):
    """Ensure provider rules refer to paths declared by the actual contract."""
    document = json.loads(pathlib.Path(spec_output).read_text(encoding="utf-8"))
    components = document.get("components", {})
    named = components.get("parameters", {}) if isinstance(components, dict) else {}
    found = set()
    for path, path_item in document.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method in ("get", "post", "put", "patch", "delete", "head", "options"):
            operation = path_item.get(method)
            if not isinstance(operation, dict) or operation.get("operationId") not in prefixes:
                continue
            operation_id = operation["operationId"]
            parameters = list(path_item.get("parameters", [])) + list(operation.get("parameters", []))
            for parameter in parameters:
                if not isinstance(parameter, dict):
                    continue
                reference = parameter.get("$ref", "")
                prefix = "#/components/parameters/"
                if isinstance(reference, str) and reference.startswith(prefix):
                    parameter = named.get(reference[len(prefix):], {})
                name = parameter.get("name", "") if isinstance(parameter, dict) else ""
                if isinstance(parameter, dict) and parameter.get("in") == "path" and name in prefixes[operation_id] and "{" + name + "}" in path:
                    found.add((operation_id, name))
    missing = sorted((operation, name) for operation, rules in prefixes.items() for name in rules
                     if (operation, name) not in found)
    if missing:
        raise ValueError("parameter prefix names an undeclared path parameter: " +
                         ", ".join(operation + "." + name for operation, name in missing))


def update_openapi_parameter_descriptions(spec_output, overrides):
    if not overrides:
        return
    path = pathlib.Path(spec_output)
    document = json.loads(path.read_text(encoding="utf-8"))
    found = set()
    components = document.get("components", {})
    named_parameters = components.get("parameters", {}) if isinstance(components, dict) else {}
    for path_item in document.get("paths", {}).values():
        if not isinstance(path_item, dict):
            continue
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId", "")
            parameters = operation.get("parameters", [])
            if not isinstance(parameters, list):
                continue
            for index, parameter in enumerate(parameters):
                if not isinstance(parameter, dict):
                    continue
                resolved = parameter
                reference = parameter.get("$ref", "")
                prefix = "#/components/parameters/"
                if isinstance(reference, str) and reference.startswith(prefix):
                    resolved = named_parameters.get(reference[len(prefix):])
                    if not isinstance(resolved, dict):
                        continue
                key = operation_id + "." + resolved.get("name", "")
                if key in overrides:
                    if resolved is not parameter:
                        resolved = copy.deepcopy(resolved)
                        resolved.update({name: value for name, value in parameter.items()
                                         if name != "$ref"})
                        parameters[index] = resolved
                    resolved["description"] = overrides[key]
                    found.add(key)
    missing = sorted(set(overrides) - found)
    if missing:
        raise SystemExit("OpenAPI parameters not found: " + ", ".join(missing))
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def validate_result_views(views):
    """Check the portable shape; the host performs authoritative validation."""
    def pointer(value):
        if not isinstance(value, str) or not value.startswith("/") or len(value) > 1024:
            return False
        position = 0
        while position < len(value):
            if value[position] == "~":
                if position + 1 == len(value) or value[position + 1] not in "01":
                    return False
                position += 1
            position += 1
        return True

    allowed = {"kind", "items_pointer", "output_pointer", "fields", "derived_fields", "max_items", "max_text_bytes", "max_bytes", "materialize", "resource_scope"}
    reserved = {"resource_ref", "dataset", "source_resource", "materialized", "representation", "pagination", "host_item_parameter", "host_item_candidates", "host_item_candidates_truncated", "result_count", "preview_omitted_items", "view_truncated", "view_invalid_fields", "materialization_reason", "mime_type", "size_bytes", "rows", "columns", "resources", "dataset_refs"}
    for operation, view in views.items():
        if not isinstance(operation, str) or not operation or not isinstance(view, dict):
            raise ValueError("result views must map operation IDs to objects")
        if set(view) - allowed or view.get("kind") not in ("object", "collection"):
            raise ValueError(f"invalid result view for {operation}")
        if view["kind"] == "collection" and view.get("items_pointer") != "" and not pointer(view.get("items_pointer")):
            raise ValueError(f"invalid items_pointer for {operation}")
        if "output_pointer" in view and (view["kind"] != "collection" or not pointer(view["output_pointer"])):
            raise ValueError(f"invalid output_pointer for {operation}")
        if view["kind"] == "object" and ("items_pointer" in view or "max_items" in view):
            raise ValueError(f"object view has collection settings for {operation}")
        if view["kind"] == "collection":
            output = view.get("output_pointer", view["items_pointer"] or "/items")
            if any(output == "/" + key or output.startswith("/" + key + "/") for key in reserved):
                raise ValueError(f"output pointer collides with host metadata for {operation}")
        fields = view.get("fields")
        if not isinstance(fields, dict) or len(fields) > 64 or any(not isinstance(k, str) or not k or len(k) > 128 or k in reserved or not pointer(v) for k, v in fields.items()):
            raise ValueError(f"invalid fields for {operation}")
        derived = view.get("derived_fields", {})
        if not isinstance(derived, dict) or len(derived) > 32:
            raise ValueError(f"invalid derived_fields for {operation}")
        for name, transform in derived.items():
            if not isinstance(name, str) or not name or len(name) > 128 or name in reserved or name in fields or not isinstance(transform, dict) or set(transform) != {"transform", "from"}:
                raise ValueError(f"invalid derived field for {operation}")
            kind, source = transform["transform"], transform["from"]
            if kind == "inverted_index_to_text" and pointer(source):
                continue
            if kind == "first_nonempty_url" and isinstance(source, list) and 0 < len(source) <= 16 and all(pointer(p) for p in source) and name + "_source" not in fields and name + "_source" not in derived:
                continue
            raise ValueError(f"invalid transform for {operation}.{name}")
        if view.get("resource_scope", "turn") not in ("turn", "session"):
            raise ValueError(f"invalid resource_scope for {operation}")
        if view.get("materialize", "resource") not in ("resource", "dataset_if_tabular"):
            raise ValueError(f"invalid materialize policy for {operation}")
        for key, maximum in (("max_items", 100), ("max_text_bytes", 32768), ("max_bytes", 65536)):
            if key in view and (type(view[key]) is not int or not 0 < view[key] <= maximum):
                raise ValueError(f"invalid {key} for {operation}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("openapi", "mcp"), required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--spec")
    parser.add_argument(
        "--spec-url",
        help="Explicit URL for the OpenAPI document when it is hosted separately from --base-url",
    )
    parser.add_argument("--spec-output")
    parser.add_argument(
        "--anonymous-public-reads",
        action="store_true",
        help=("mark GET, HEAD and OPTIONS operations public in the generated copy; "
              "use only after the operator has verified anonymous read access"),
    )
    parser.add_argument("--allowed-operation", action="append", default=[],
                        help="Explicit OpenAPI operationId to expose (repeatable; enables include policy)")
    parser.add_argument("--default-projection", action="append", default=[],
                        metavar="OPERATION_ID=FIELDS",
                        help="Host default for a declared select parameter (repeatable)")
    parser.add_argument("--required-any", action="append", default=[], metavar="OPERATION_ID=PARAMETER,...",
                        help="Require at least one declared parameter for an operation")
    parser.add_argument("--parameter-description", action="append", default=[],
                        metavar="OPERATION_ID.PARAMETER=TEXT",
                        help="Override one operation's inline parameter description (repeatable)")
    parser.add_argument("--default-page-size", type=int)
    parser.add_argument("--max-page-size", type=int)
    parser.add_argument("--result-views-file", metavar="PATH",
                        help="JSON object mapping operation IDs to host result-view policies")
    parser.add_argument("--parameter-prefixes-file", metavar="PATH",
                        help="JSON object mapping operation IDs to bounded path parameter prefix rules")
    parser.add_argument("--required", action="store_true",
                        help="Fail host startup if this provider cannot be loaded")
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
    if args.required:
        provider["required"] = True
    if args.prefix:
        provider["prefix"] = args.prefix
    provider["auth"] = {"type": args.auth_type}
    if args.auth_type == "bearer":
        provider["auth"]["token_env"] = args.token_env

    if args.type == "openapi":
        if not args.base_url:
            parser.error("OpenAPI providers require --base-url")
        if args.default_page_size is not None and args.default_page_size < 1:
            parser.error("--default-page-size must be positive")
        if args.max_page_size is not None and args.max_page_size < 1:
            parser.error("--max-page-size must be positive")
        if args.default_page_size is not None and args.max_page_size is not None and args.default_page_size > args.max_page_size:
            parser.error("--default-page-size cannot exceed --max-page-size")
        provider["base_url"] = args.base_url
        spec = args.spec
        if not spec:
            spec = args.spec_output or (args.id + "-openapi.json")
            if args.spec_url:
                discovered = fetch_openapi_url((args.spec_url,), spec)
            else:
                discovered = fetch_openapi(args.base_url, spec)
            print("discovered OpenAPI document: " + discovered, file=sys.stderr)
        if args.anonymous_public_reads:
            make_public_read_operations_anonymous(spec, set(args.allowed_operation))
            print("marked safe operations public in generated OpenAPI copy", file=sys.stderr)
        update_openapi_parameter_descriptions(
            spec, parse_operation_values(args.parameter_description, "--parameter-description"))
        provider["spec_path"] = spec
        projections = parse_operation_values(args.default_projection, "--default-projection")
        required_any = parse_operation_values(args.required_any, "--required-any")
        if args.allowed_operation and not set(required_any).issubset(args.allowed_operation):
            parser.error("required-any operation must be in the allowlist")
        views = {}
        if args.result_views_file:
            try:
                views = json.loads(pathlib.Path(args.result_views_file).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                parser.error(f"cannot read result views: {exc}")
            if not isinstance(views, dict):
                parser.error("result views must map operation IDs to objects")
            try:
                validate_result_views(views)
            except ValueError as exc:
                parser.error(str(exc))
            if args.allowed_operation and not set(views).issubset(args.allowed_operation):
                parser.error("result view operation must be in the allowlist")
        prefixes = {}
        if args.parameter_prefixes_file:
            try:
                prefixes = json.loads(pathlib.Path(args.parameter_prefixes_file).read_text(encoding="utf-8"))
                if not isinstance(prefixes, dict) or len(prefixes) > 256:
                    raise ValueError("parameter prefixes must map operation IDs to rule objects")
                for operation, rules in prefixes.items():
                    if not isinstance(operation, str) or not operation or not isinstance(rules, dict) or len(rules) > 32:
                        raise ValueError("invalid parameter prefix operation")
                    for name, rule in rules.items():
                        if (not isinstance(name, str) or not name or len(name) > 128 or
                                not isinstance(rule, dict) or set(rule) != {"if_starts_with", "prepend"} or
                                not isinstance(rule["if_starts_with"], str) or not 0 < len(rule["if_starts_with"]) <= 128 or
                                not isinstance(rule["prepend"], str) or not 0 < len(rule["prepend"]) <= 512):
                            raise ValueError(f"invalid parameter prefix rule for {operation}.{name}")
                if args.allowed_operation and not set(prefixes).issubset(args.allowed_operation):
                    raise ValueError("parameter prefix operation must be in the allowlist")
                validate_path_parameter_prefixes(spec, prefixes)
            except (OSError, ValueError) as exc:
                parser.error(f"invalid parameter prefixes: {exc}")
        if args.allowed_operation or projections or views or required_any or prefixes:
            operations = {operation_id: {"enabled": True} for operation_id in args.allowed_operation}
            for operation_id, values in required_any.items():
                parameters = [value.strip() for value in values.split(",")]
                if any(not value for value in parameters) or len(set(parameters)) != len(parameters):
                    parser.error("--required-any requires distinct nonempty parameter names")
                operations.setdefault(operation_id, {"enabled": True})["required_any_parameters"] = parameters
            for operation_id, view in views.items():
                operations.setdefault(operation_id, {"enabled": True})["result_view"] = view
            for operation_id, rules in prefixes.items():
                operations.setdefault(operation_id, {"enabled": True})["parameter_prefixes"] = rules
            for operation_id, projection in projections.items():
                operations.setdefault(operation_id, {"enabled": True})
                operations[operation_id]["default_projection"] = projection
            provider["policy"] = {
                "access": "read_only",
                "exposure": "include",
                "operations": operations,
            }
        limits = {}
        if args.default_page_size is not None:
            limits["default_page_size"] = args.default_page_size
        if args.max_page_size is not None:
            limits["max_page_size"] = args.max_page_size
        if limits:
            provider["limits"] = limits
    else:
        if args.result_views_file or args.parameter_prefixes_file:
            parser.error("result view and parameter prefix files apply only to OpenAPI providers")
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
