export function structuredToolResult(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("The daemon returned no tool result.");
  }
  const envelope = payload as Record<string, unknown>;
  const tool = envelope.tool && typeof envelope.tool === "object" && !Array.isArray(envelope.tool)
    ? envelope.tool as Record<string, unknown> : envelope;
  const error = tool.error && typeof tool.error === "object" && !Array.isArray(tool.error)
    ? tool.error as Record<string, unknown> : {};
  if (tool.ok === false || Object.keys(error).length > 0) {
    throw new Error(String(error.safe_summary ?? tool.content_summary ?? envelope.error ?? "Tool execution failed."));
  }

  const structured = tool.structured_content && typeof tool.structured_content === "object" && !Array.isArray(tool.structured_content)
    ? tool.structured_content as Record<string, unknown>
    : tool.structuredContent && typeof tool.structuredContent === "object" && !Array.isArray(tool.structuredContent)
      ? tool.structuredContent as Record<string, unknown>
      : tool;
  const nestedError = structured.error && typeof structured.error === "object" && !Array.isArray(structured.error)
    ? structured.error as Record<string, unknown> : {};
  if (structured.ok === false || Object.keys(nestedError).length > 0) {
    throw new Error(String(nestedError.safe_summary ?? structured.summary ?? "Tool execution failed."));
  }
  return structured.result && typeof structured.result === "object" && !Array.isArray(structured.result)
    ? structured.result as Record<string, unknown>
    : structured;
}
