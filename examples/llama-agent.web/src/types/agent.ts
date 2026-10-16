export type AgentEvent = {
  type?: string;
  event_type?: string;
  message_type?: string;
  sequence?: number;
  timestamp?: string | number;
  created_at?: string | number;
  client_received_at?: number;
  request_id?: string;
  turn_id?: string;
  session_id?: string;
  operation_id?: string;
  tool_name?: string;
  app_context_available?: boolean;
  provider_id?: string;
  server_tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_result?: unknown;
  detail?: string;
  response?: string;
  error?: string;
  [key: string]: unknown;
};

export type DaemonStatus = {
  ok?: boolean;
  status?: string;
  state?: string;
  active_turn_id?: string;
  readiness?: {
    tools?: DaemonToolStatus[];
    providers?: DaemonProviderStatus[];
    warnings?: string[];
    tool_profile?: string;
    stores?: Record<string, string>;
    [key: string]: unknown;
  };
  metrics?: Record<string, number>;
  session_keys?: unknown[];
  [key: string]: unknown;
};

export type DaemonToolStatus = {
  name: string;
  description?: string;
  source?: string;
  state?: "active" | "degraded" | "disabled" | string;
  input_schema?: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
  read_only?: boolean;
  app_read_only?: boolean;
  policy_gated?: boolean;
  provider_id?: string;
  server_tool_name?: string;
  _meta?: { ui?: { resourceUri?: string; visibility?: string[]; [key: string]: unknown } };
};

export type DaemonProviderStatus = {
  id?: string;
  name?: string;
  status?: string;
};

export type ChatMessage = {
  id: string;
  turnId: string;
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
};

export type TurnRequest = {
  prompt: string;
  session_id: string;
  namespace_id: string;
  project_id: string;
  turn_id: string;
  mode: string;
  n_predict: number;
  include_summary: boolean;
  resource_refs?: string[];
};

export type UploadedResource = {
  name: string;
  mime_type: string;
  text?: string;
  bytes_base64?: string;
  scope: "session";
  description?: string;
  session_id?: string;
  namespace_id?: string;
  project_id?: string;
  uri?: string;
};

export type TurnResult = {
  ok?: boolean;
  response?: string;
  error?: string;
  turn_id?: string;
  [key: string]: unknown;
};

export type ToolCallResult = {
  ok?: boolean;
  tool?: {
    ok?: boolean;
    tool_name?: string;
    content_summary?: string;
    structured_content?: unknown;
    resource_refs?: Array<{ uri?: string; name?: string; mime_type?: string }>;
    dataset_refs?: unknown[];
    error?: { code?: string; class?: string; retryable?: boolean; safe_summary?: string };
  };
  error?: string;
};

export type PutResourceResult = {
  resource?: { uri?: string; name?: string; mime_type?: string; [key: string]: unknown };
  [key: string]: unknown;
};
