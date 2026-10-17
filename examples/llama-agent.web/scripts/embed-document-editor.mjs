import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const repo = resolve(root, "../..");
const template = await readFile(resolve(root, "src/mcp-app/document-editor.html"), "utf8");
const bundle = await readFile(resolve(root, "dist/document-editor.js"), "utf8");
const css = await readFile(resolve(root, "dist/document-editor.css.css"), "utf8");
const escapeHtml = (value) => value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const thirdParty = [
  ["Notice and attributions", resolve(repo, "docs/third-party/NOTICE.md")],
  ["TOAST UI Editor 3.2.2 · MIT", resolve(repo, "docs/third-party/TOAST-UI-MIT.txt")],
  ["DOMPurify 2.5.9 · Apache-2.0 or MPL-2.0", resolve(root, "node_modules/dompurify/LICENSE")],
  ...["prosemirror-commands", "prosemirror-history", "prosemirror-inputrules", "prosemirror-keymap", "prosemirror-model", "prosemirror-state", "prosemirror-transform", "prosemirror-view", "orderedmap", "rope-sequence", "w3c-keyname"]
    .map((name) => [`${name} · MIT`, resolve(root, `node_modules/${name}/LICENSE`)]),
];
const noticeHtml = (await Promise.all(thirdParty.map(async ([title, path]) =>
  `<details><summary>${escapeHtml(title)}</summary><pre>${escapeHtml(await readFile(path, "utf8"))}</pre></details>`))).join("");
if (!template.includes("__LLAMA_AGENT_DOCUMENT_EDITOR_BUNDLE__") || !template.includes("__LLAMA_AGENT_DOCUMENT_EDITOR_CSS__") || !template.includes("__LLAMA_AGENT_THIRD_PARTY_NOTICES__")) throw new Error("document app template is missing a bundle marker");
const inlineBundle = bundle.replace(/<\/script/gi, "<\\/script");
const html = template.replace("__LLAMA_AGENT_THIRD_PARTY_NOTICES__", () => noticeHtml).replace("__LLAMA_AGENT_DOCUMENT_EDITOR_CSS__", () => css).replace("__LLAMA_AGENT_DOCUMENT_EDITOR_BUNDLE__", () => inlineBundle);
const bytes = Buffer.from(html, "utf8");
if (bytes.length > 2 * 1024 * 1024) throw new Error("document editor exceeds the MCP App host's 2 MiB limit");
await writeFile(resolve(root, "dist/document-editor.html"), html, "utf8");
let literal = '"';
for (const byte of bytes) {
  if (byte === 0x22) literal += '\\"';
  else if (byte === 0x3f) literal += '\\?';
  else if (byte === 0x5c) literal += '\\\\';
  else if (byte === 0x0a) literal += '\\n';
  else if (byte === 0x0d) literal += '\\r';
  else if (byte === 0x09) literal += '\\t';
  else if (byte < 0x20 || byte >= 0x7f) literal += `\\${byte.toString(8).padStart(3, "0")}`;
  else literal += String.fromCharCode(byte);
}
literal += '"';
const header = `#pragma once\nstatic constexpr char LLAMA_AGENT_DOCUMENT_EDITOR_HTML[] = ${literal};\n`;
await writeFile(resolve(root, "dist/document-editor-html.h"), header, "utf8");
