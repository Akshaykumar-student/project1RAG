export interface Citation {
  label: string;
  source_type: "file" | "web";
  filename?: string | null;
  url?: string | null;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  source: "documentation" | "web" | "general" | "unavailable";
  media: ChatMedia[];
  media_error?: MediaError | null;
}

export interface MediaError {
  code: "missing_api_key" | "insufficient_quota" | "model_access_denied" | "rate_limit" | "safety_rejected" | "generation_timeout" | "invalid_response" | "network_error" | "provider_error";
  retryable: boolean;
}

export interface ChatMedia {
  id: number;
  type: "image" | "video";
  mime_type: string;
  alt: string;
}

export interface ChatAttachment {
  filename: string;
  mime_type: string;
  encoding: "utf8" | "base64";
  size: number;
  content: string;
}

export interface StoredAttachment {
  id: number;
  filename: string;
  mime_type: string;
  size: number;
  preview_type: "pdf" | "image" | "text" | "document";
  /** Present only on an optimistic message while the server analyzes the file. */
  local_encoding?: "utf8" | "base64";
  local_content?: string;
  local_url?: string;
}

export interface TemporaryUpload {
  id: string;
  filename: string;
  mime_type: string;
  size: number;
  preview_type: "pdf" | "image" | "text" | "document";
  expires_at: number;
}

export interface StoredMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  source: "user" | "documentation" | "web" | "general" | "unavailable";
  created_at: number;
  media: ChatMedia[];
  attachments?: StoredAttachment[];
  media_error?: MediaError | null;
}

export interface BoardSummary {
  id: number;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface BoardDetail extends Omit<BoardSummary, "message_count"> {
  messages: StoredMessage[];
}

export interface SearchResult {
  board_id: number;
  board_title: string;
  message_id: number;
  role: "user" | "assistant";
  snippet: string;
  created_at: number;
}

export interface User {
  id: number;
  email: string;
  name: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: "bearer";
  user: User;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

function authHeaders(token: string): Record<string, string> {
  return { "Content-Type": "application/json", Authorization: `Bearer ${token}` };
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(formatApiError(body.detail));
  }
  return response.json();
}

export function formatApiError(detail: unknown): string {
  const fallback = "FlowDesk Support is unavailable. Please try again.";
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== "object") return String(item);
      const record = item as { loc?: unknown; msg?: unknown };
      const location = Array.isArray(record.loc) ? record.loc.slice(1).join(".") : "";
      const message = typeof record.msg === "string" ? record.msg : "Invalid value";
      return location ? `${location}: ${message}` : message;
    }).filter(Boolean);
    const validationMessage = messages.join("; ");
    const attachmentMismatch = messages.some((message) => message.startsWith("attachments."));
    return attachmentMismatch
      ? `The frontend and backend attachment formats do not match. Restart both servers and attach the file again. ${validationMessage}`
      : validationMessage || fallback;
  }
  if (detail && typeof detail === "object") {
    const record = detail as { message?: unknown; detail?: unknown };
    if (typeof record.message === "string") return record.message;
    if (typeof record.detail === "string") return record.detail;
  }
  return fallback;
}

export async function authenticate(
  mode: "login" | "register",
  email: string,
  password: string,
  name?: string,
): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/api/auth/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, ...(mode === "register" ? { name } : {}) }),
  });
  return parseResponse(response);
}

export async function requestPasswordReset(email: string): Promise<string> {
  const response = await fetch(`${API_BASE}/api/auth/forgot-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  return (await parseResponse<{ message: string }>(response)).message;
}

export async function resetPassword(token: string, password: string): Promise<string> {
  const response = await fetch(`${API_BASE}/api/auth/reset-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, password }),
  });
  return (await parseResponse<{ message: string }>(response)).message;
}

export async function listBoards(token: string): Promise<BoardSummary[]> {
  return parseResponse(await fetch(`${API_BASE}/api/boards`, { headers: authHeaders(token) }));
}

export async function createBoard(token: string, title = "New chat"): Promise<BoardSummary> {
  return parseResponse(await fetch(`${API_BASE}/api/boards`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ title }),
  }));
}

export async function getBoard(token: string, boardId: number): Promise<BoardDetail> {
  return parseResponse(
    await fetch(`${API_BASE}/api/boards/${boardId}`, { headers: authHeaders(token) }),
  );
}

export async function searchMessages(
  token: string,
  query: string,
  signal?: AbortSignal,
): Promise<SearchResult[]> {
  const params = new URLSearchParams({ q: query });
  return parseResponse(await fetch(`${API_BASE}/api/search?${params}`, {
    headers: authHeaders(token),
    signal,
  }));
}

export async function renameBoard(
  token: string,
  boardId: number,
  title: string,
): Promise<BoardDetail> {
  return parseResponse(await fetch(`${API_BASE}/api/boards/${boardId}`, {
    method: "PATCH",
    headers: authHeaders(token),
    body: JSON.stringify({ title }),
  }));
}

export async function deleteBoard(token: string, boardId: number): Promise<void> {
  const response = await fetch(`${API_BASE}/api/boards/${boardId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!response.ok) await parseResponse(response);
}

export async function askFlowDesk(
  question: string,
  token: string,
  boardId: number,
  attachments: ChatAttachment[] = [],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ question, board_id: boardId, attachments }),
    signal,
  });
  return parseResponse(response);
}

export async function streamFlowDesk(
  question: string,
  token: string,
  boardId: number,
  attachmentIds: string[] = [],
  signal?: AbortSignal,
  onDelta?: (text: string) => void,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ question, board_id: boardId, attachment_ids: attachmentIds }),
    signal,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(formatApiError(body.detail));
  }
  // Supports lightweight fetch mocks and older compatible backends in tests.
  if (!response.body) return response.json();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed: ChatResponse | null = null;

  const consumeLine = (line: string) => {
    if (!line.trim()) return;
    const event = JSON.parse(line) as ({ type: "delta"; text: string }
      | ({ type: "complete" } & ChatResponse)
      | { type: "error"; message: string });
    if (event.type === "delta") onDelta?.(event.text);
    else if (event.type === "error") throw new Error(event.message);
    else completed = event;
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) consumeLine(line);
    if (done) break;
  }
  if (buffer.trim()) consumeLine(buffer);
  if (!completed) throw new Error("The AI response ended before it was complete.");
  return completed;
}


export function uploadAttachment(
  token: string,
  file: File,
  onProgress: (percent: number) => void,
): Promise<TemporaryUpload> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE}/api/uploads`);
    request.setRequestHeader("Authorization", `Bearer ${token}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    };
    request.onerror = () => reject(new Error("The file upload could not connect to the server."));
    request.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    request.onload = () => {
      let body: unknown = {};
      try { body = JSON.parse(request.responseText); } catch { /* Use the safe fallback below. */ }
      if (request.status >= 200 && request.status < 300) {
        resolve(body as TemporaryUpload);
      } else {
        reject(new Error(formatApiError((body as { detail?: unknown }).detail)));
      }
    };
    const form = new FormData();
    form.append("file", file, file.name);
    request.send(form);
  });
}

export async function deleteTemporaryUpload(token: string, uploadId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/uploads/${encodeURIComponent(uploadId)}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!response.ok && response.status !== 404) await parseResponse(response);
}

export async function getMediaBlob(
  token: string,
  mediaId: number,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/media/${mediaId}`, {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  });
  if (!response.ok) throw new Error("Could not load this generated image.");
  return response.blob();
}

export async function getAttachmentBlob(
  token: string,
  attachmentId: number,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/attachments/${attachmentId}`, {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  });
  if (!response.ok) throw new Error(formatApiError((await response.json().catch(() => ({}))).detail));
  return response.blob();
}

export async function getAttachmentPreview(
  token: string,
  attachmentId: number,
  signal?: AbortSignal,
): Promise<{ filename: string; content: string }> {
  const response = await fetch(`${API_BASE}/api/attachments/${attachmentId}/preview`, {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  });
  return parseResponse(response);
}
