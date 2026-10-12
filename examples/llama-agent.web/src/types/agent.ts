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

export type PutResourceResult = {
  resource?: { uri?: string; name?: string; mime_type?: string; [key: string]: unknown };
  [key: string]: unknown;
};
