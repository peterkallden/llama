#!/usr/bin/env python3
"""Offline contract tests for the provider configuration generator."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


class ProviderBootstrapTest(unittest.TestCase):
    def test_openapi_allowlist_and_public_read_override(self):
        repository = pathlib.Path(sys.argv[1]).resolve()
        script = repository / "scripts/agent-provider-bootstrap.py"
        with tempfile.TemporaryDirectory(prefix="agent-provider-bootstrap-") as temp:
            root = pathlib.Path(temp)
            spec_path = root / "spec.json"
            output_path = root / "provider.json"
            fixture = {
                "openapi": "3.1.0",
                "security": [{"apiKey": []}],
                "paths": {
                    "/works": {
                        "get": {
                            "operationId": "listWorks",
                            "parameters": [
                                {"name": "filter", "in": "query", "schema": {"type": "string"}},
                                {"$ref": "#/components/parameters/search"},
                                {"$ref": "#/components/parameters/api_key"},
                            ],
                        },
                        "post": {
                            "operationId": "createWork",
                            "parameters": [{"$ref": "#/components/parameters/api_key"}],
                        },
                    }
                },
                "components": {
                    "parameters": {
                        "search": {"name": "search", "in": "query", "description": "Shared search",
                                   "schema": {"type": "string"}},
                        "api_key": {"name": "api_key", "in": "query", "required": True,
                                     "schema": {"type": "string"}},
                    }
                },
            }
            spec_path.write_text(json.dumps(fixture), encoding="utf-8")
            subprocess.run([
                sys.executable, str(script), "--type", "openapi", "--id", "example",
                "--base-url", "https://api.example.test", "--spec", str(spec_path),
                "--output", str(output_path), "--required", "--anonymous-public-reads",
                "--allowed-operation", "listWorks",
                "--default-projection", "listWorks=id,display_name,publication_year",
                "--default-page-size", "5", "--max-page-size", "10",
                "--parameter-description", "listWorks.filter=Year, author, institution or topic filters",
                "--parameter-description", "listWorks.search=Search work titles and abstracts",
            ], check=True, capture_output=True, text=True)

            generated_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            get_operation = generated_spec["paths"]["/works"]["get"]
            self.assertEqual(get_operation["security"], [])
            self.assertEqual(len(get_operation["parameters"]), 2)
            self.assertEqual(get_operation["parameters"][0]["description"],
                             "Year, author, institution or topic filters")
            self.assertEqual(get_operation["parameters"][1]["description"],
                             "Search work titles and abstracts")
            self.assertEqual(generated_spec["components"]["parameters"]["search"]["description"],
                             "Shared search")
            self.assertNotIn("security", generated_spec["paths"]["/works"]["post"])
            self.assertIn({"$ref": "#/components/parameters/api_key"},
                          generated_spec["paths"]["/works"]["post"]["parameters"])
            self.assertEqual(generated_spec["security"], [{"apiKey": []}])

            provider = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(provider["required"])
            self.assertEqual(provider["policy"]["access"], "read_only")
            self.assertEqual(provider["policy"]["exposure"], "include")
            self.assertEqual(list(provider["policy"]["operations"]), ["listWorks"])
            self.assertEqual(provider["policy"]["operations"]["listWorks"]["default_projection"],
                             "id,display_name,publication_year")
            self.assertEqual(provider["limits"], {"default_page_size": 5, "max_page_size": 10})


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
