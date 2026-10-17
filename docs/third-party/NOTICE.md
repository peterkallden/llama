# Third-party notices for the document editor

The embedded document-editor app bundles the following pinned runtime
dependencies. These notices apply to the distributed web bundle; the project
code remains MIT-licensed under the repository's `LICENSE`.

| Package | Version | License | Attribution |
| --- | --- | --- | --- |
| `@toast-ui/editor` | 3.2.2 | MIT | NHN Cloud Corp. |
| `dompurify` | 2.5.9 | Apache-2.0 option of `Apache-2.0 OR MPL-2.0` | Mario Heiderich / Cure53 |
| `prosemirror-commands` | 1.7.2 | MIT | Marijn Haverbeke and contributors |
| `prosemirror-history` | 1.5.1 | MIT | Marijn Haverbeke and contributors |
| `prosemirror-inputrules` | 1.5.1 | MIT | Marijn Haverbeke and contributors |
| `prosemirror-keymap` | 1.2.3 | MIT | Marijn Haverbeke and contributors |
| `prosemirror-model` | 1.25.12 | MIT | Marijn Haverbeke and contributors |
| `prosemirror-state` | 1.4.4 | MIT | Marijn Haverbeke and contributors |
| `prosemirror-transform` | 1.12.2 | MIT | Marijn Haverbeke and contributors |
| `prosemirror-view` | 1.42.6 | MIT | Marijn Haverbeke and contributors |
| `orderedmap` | 2.1.1 | MIT | Marijn Haverbeke |
| `rope-sequence` | 1.3.4 | MIT | Marijn Haverbeke |
| `w3c-keyname` | 2.2.8 | MIT | Marijn Haverbeke |

The MIT license text for TOAST UI is in `TOAST-UI-MIT.txt`. The ProseMirror
packages use the MIT license text and copyright notices included in their npm
package metadata/source distributions. DOMPurify's complete dual-license
notice (including both Apache-2.0 and MPL-2.0 texts) is installed from its
locked package. cmark-gfm's BSD-2-Clause/MIT notices are installed from the
pinned upstream `COPYING` file. Its upstream specification/test corpus is not
built or distributed.

Dependency versions are locked in `examples/llama-agent.web/package-lock.json`.
