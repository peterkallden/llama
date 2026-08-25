export type AgentEvent = {
  type?: string;
  event_type?: string;
  message_type?: string;
  sequence?: number;
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
  [key: string]: unknown;
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
  text: string;
  scope: "session";
  description?: string;
  session_id?: string;
  namespace_id?: string;
  project_id?: string;
};

export type PutResourceResult = {
  resource?: { uri?: string; name?: string; mime_type?: string; [key: string]: unknown };
  [key: string]: unknown;
};
