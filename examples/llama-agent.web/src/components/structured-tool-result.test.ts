import { describe, expect, it } from "vitest";
import { structuredToolResult } from "./structured-tool-result";

describe("structured tool result normalization", () => {
  it("unwraps daemon tool and structured success envelopes", () => {
    expect(structuredToolResult({
      ok: true,
      tool: {
        ok: true,
        structured_content: {
          ok: true,
          result: { document_uri: "agent-resource://resource/r1", revision: 1 },
          summary: "Imported",
        },
      },
    })).toEqual({ document_uri: "agent-resource://resource/r1", revision: 1 });
  });

  it("accepts a direct structured result", () => {
    expect(structuredToolResult({ tool: { ok: true, structured_content: { id: "d1" } } }))
      .toEqual({ id: "d1" });
  });

  it("surfaces safe tool failures", () => {
    expect(() => structuredToolResult({
      ok: false,
      tool: { ok: false, error: { safe_summary: "Import is unavailable." } },
    })).toThrow("Import is unavailable.");
  });
});
