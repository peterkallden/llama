import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const template = await readFile(resolve(root, "src/mcp-app/investigation-workspace.html"), "utf8");
const bundle = await readFile(resolve(root, "dist/investigation-workspace.js"), "utf8");
const css = await readFile(resolve(root, "dist/investigation-workspace.css.css"), "utf8");
const license = await readFile(resolve(root, "node_modules/cytoscape/LICENSE"), "utf8");
if (!template.includes("__INVESTIGATION_WORKSPACE_CSS__") || !template.includes("__INVESTIGATION_WORKSPACE_BUNDLE__")) {
  throw new Error("investigation workspace template markers are missing");
}
const safeScript = bundle.replace(/<\/script/gi, "<\\/script");
const html = template
  .replace("__INVESTIGATION_WORKSPACE_CSS__", css)
  .replace("__INVESTIGATION_WORKSPACE_BUNDLE__", safeScript)
  .replace("</body>", `<details><summary>Third-party notice · Cytoscape.js MIT</summary><pre>${license.replaceAll("&", "&amp;").replaceAll("<", "&lt;")}</pre></details></body>`);
const bytes = Buffer.from(html, "utf8");
if (bytes.length > 2 * 1024 * 1024) throw new Error("investigation workspace exceeds the MCP App host's 2 MiB limit");
await writeFile(resolve(root, "dist/investigation-workspace.html"), html, "utf8");
let literal = '"';
for (const byte of bytes) {
  if (byte === 0x22) literal += '\\"';
  else if (byte === 0x3f) literal += "\\?";
  else if (byte === 0x5c) literal += "\\\\";
  else if (byte === 0x0a) literal += "\\n";
  else if (byte === 0x0d) literal += "\\r";
  else if (byte === 0x09) literal += "\\t";
  else if (byte < 0x20 || byte >= 0x7f) literal += `\\${byte.toString(8).padStart(3, "0")}`;
  else literal += String.fromCharCode(byte);
}
literal += '"';
await writeFile(resolve(root, "dist/investigation-workspace-html.h"),
  `#pragma once\nstatic constexpr char LLAMA_AGENT_INVESTIGATION_WORKSPACE_HTML[] = ${literal};\n`, "utf8");
