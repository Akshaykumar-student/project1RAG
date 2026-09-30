import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { formatApiError } from "./api";

const board = { id: 1, title: "New chat", created_at: 1, updated_at: 1, message_count: 0 };

function response(body: unknown, ok = true) {
  return Promise.resolve({ ok, json: async () => body });
}

function mockBoardApi(chatError?: string) {
  let answered = false;
  return vi.fn((url: string, options?: RequestInit) => {
    if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) {
      return response([board]);
    }
    if (url.endsWith("/api/boards/1")) {
      return response({
        ...board,
        messages: answered ? [
          { id: 1, role: "user", content: "When does it expire?", citations: [], source: "user", created_at: 1 },
          { id: 2, role: "assistant", content: "Invites expire after seven days.", citations: [{ source_type: "file", filename: "02-invite-team-members.md", label: "Invite Team Members" }], source: "documentation", created_at: 2 },
        ] : [],
      });
    }
    if (url.endsWith("/api/chat/stream")) {
      if (chatError) return response({ detail: chatError }, false);
      answered = true;
      return response({ answer: "Invites expire after seven days.", citations: [], source: "general" });
    }
    return response(board);
  });
}

afterEach(() => vi.restoreAllMocks());

beforeEach(() => {
  class MockXMLHttpRequest {
    upload: { onprogress: ((event: ProgressEvent) => void) | null } = { onprogress: null };
    status = 201;
    responseText = "";
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onabort: (() => void) | null = null;
    open() { /* Test transport. */ }
    setRequestHeader() { /* Test transport. */ }
    send(body: FormData) {
      const file = body.get("file") as File;
      this.responseText = JSON.stringify({
        id: `upload-${file.name}`,
        filename: file.name,
        mime_type: file.type,
        size: file.size,
        preview_type: "text",
        expires_at: 9999999999,
      });
      this.upload.onprogress?.({ lengthComputable: true, loaded: file.size, total: file.size } as ProgressEvent);
      queueMicrotask(() => this.onload?.());
    }
  }
  vi.stubGlobal("XMLHttpRequest", MockXMLHttpRequest);
  Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:attachment") });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  sessionStorage.setItem("flowdesk_token", "test-token");
  sessionStorage.setItem("flowdesk_user", JSON.stringify({ id: 1, email: "test@example.com", name: "Test User" }));
});

describe("FlowDesk chat boards", () => {
  it("formats structured validation errors instead of object text", () => {
    expect(formatApiError([{ loc: ["body", "attachments", 0, "size"], msg: "Field required" }]))
      .toContain("attachments.0.size: Field required");
    expect(formatApiError({ unexpected: true })).not.toContain("[object Object]");
  });

  it("loads a board and shows sample questions", async () => {
    vi.stubGlobal("fetch", mockBoardApi());
    render(<App />);
    expect(await screen.findByText("How do I invite a teammate?")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "New chat" })).toBeInTheDocument();
  });

  it("collapses and expands the sidebar", async () => {
    vi.stubGlobal("fetch", mockBoardApi());
    const { container } = render(<App />);
    const collapse = await screen.findByLabelText("Collapse sidebar");
    await userEvent.click(collapse);
    expect(container.querySelector(".boards-layout")).toHaveClass("sidebar-collapsed");
    const expand = screen.getByLabelText("Expand sidebar");
    expect(expand).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(expand);
    expect(container.querySelector(".boards-layout")).not.toHaveClass("sidebar-collapsed");
  });

  it("toggles recent chats", async () => {
    vi.stubGlobal("fetch", mockBoardApi());
    render(<App />);
    const recents = await screen.findByRole("button", { name: "Recents" });
    expect(screen.getByRole("navigation", { name: "Recent chats" })).toBeInTheDocument();
    await userEvent.click(recents);
    expect(screen.queryByRole("navigation", { name: "Recent chats" })).not.toBeInTheDocument();
    await userEvent.click(recents);
    expect(screen.getByRole("navigation", { name: "Recent chats" })).toBeInTheDocument();
  });

  it("searches saved messages and opens the matching message", async () => {
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(),
    });
    const matchingMessage = { id: 44, role: "assistant", content: "Refunds take five days.", citations: [], source: "general", created_at: 2 };
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url.includes("/api/search?")) return response([{ board_id: 1, board_title: "Billing help", message_id: 44, role: "assistant", snippet: "Refunds take five days.", created_at: 2 }]);
      if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) return response([board]);
      if (url.endsWith("/api/boards/1")) return response({ ...board, title: "Billing help", messages: [matchingMessage] });
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Search chats"), "refund");
    const resultType = await screen.findByText("FlowDesk answer");
    await userEvent.click(resultType.closest("button")!);
    await waitFor(() => expect(document.getElementById("message-44")).toHaveClass("message-highlighted"));
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: originalScrollIntoView,
    });
  });

  it("stores and renders an answer with its citation", async () => {
    vi.stubGlobal("fetch", mockBoardApi());
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Ask FlowDesk Support"), "When does it expire?");
    await userEvent.click(screen.getByLabelText("Send question"));
    expect(await screen.findByText("Invites expire after seven days.")).toBeInTheDocument();
    expect(screen.getByText("Invite Team Members")).toBeInTheDocument();
  });

  it("loads generated images inline with authentication", async () => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:generated-image"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) {
        return response([board]);
      }
      if (url.endsWith("/api/boards/1")) {
        return response({
          ...board,
          messages: [{
            id: 2,
            role: "assistant",
            content: "Here is the image I generated for you.",
            citations: [{ source_type: "web", url: "https://example.com/image", label: "Hidden link" }],
            source: "general",
            created_at: 2,
            media: [
              { id: 7, type: "image", mime_type: "image/png", alt: "A red fox" },
              { id: 8, type: "video", mime_type: "video/mp4", alt: "A running fox" },
            ],
          }],
        });
      }
      if (url.endsWith("/api/media/7")) {
        return Promise.resolve({ ok: true, blob: async () => new Blob(["image"]) });
      }
      if (url.endsWith("/api/media/8")) {
        return Promise.resolve({ ok: true, blob: async () => new Blob(["video"]) });
      }
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(await screen.findByAltText("A red fox")).toHaveAttribute("src", "blob:generated-image");
    expect(await screen.findByLabelText("A running fox")).toHaveAttribute("controls");
    expect(screen.getByLabelText("A running fox")).toHaveAttribute("playsinline");
    expect(screen.queryByText("Hidden link")).not.toBeInTheDocument();
    const mediaCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/media/7"));
    expect((mediaCall?.[1]?.headers as Record<string, string>).Authorization).toBe("Bearer test-token");
  });

  it("shows backend errors", async () => {
    vi.stubGlobal("fetch", mockBoardApi("Knowledge base is not configured."));
    render(<App />);
    await userEvent.click(await screen.findByText("How can I fix error FD-403?"));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("not configured"));
  });

  it("restores the draft and removes its optimistic message after an attachment failure", async () => {
    vi.stubGlobal("fetch", mockBoardApi("PDF analysis is not installed."));
    render(<App />);
    const file = new File(["resume"], "resume.txt", { type: "text/plain" });
    await userEvent.upload(await screen.findByLabelText("Attach files"), file);
    const input = screen.getByLabelText("Ask FlowDesk Support");
    await userEvent.type(input, "Read my resume");
    await userEvent.click(screen.getByLabelText("Send question"));
    await waitFor(() => expect(input).toHaveValue("Read my resume"));
    expect(screen.getByText("resume.txt", { exact: false })).toBeInTheDocument();
    expect(screen.queryByText("Attached: resume.txt", { exact: false })).not.toBeInTheDocument();
  });

  it("opens an attachment while the response is still being generated", async () => {
    const pendingChat = new Promise(() => undefined);
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) {
        return response([board]);
      }
      if (url.endsWith("/api/boards/1")) return response({ ...board, messages: [] });
      if (url.endsWith("/api/chat/stream")) return pendingChat;
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    const file = new File(["Project: Customer support RAG"], "resume.txt", { type: "text/plain" });
    await userEvent.upload(await screen.findByLabelText("Attach files"), file);
    await userEvent.type(screen.getByLabelText("Ask FlowDesk Support"), "Read this file");
    await userEvent.click(screen.getByLabelText("Send question"));
    await userEvent.click(await screen.findByRole("button", { name: "Open resume.txt" }));

    expect(screen.getByRole("dialog", { name: "Preview resume.txt" })).toBeInTheDocument();
    expect(screen.getByText("Project: Customer support RAG")).toBeInTheDocument();
    expect(screen.getByText(/Analyzing/)).toBeInTheDocument();
  });

  it("retries a temporary media failure with the original prompt", async () => {
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) {
        return response([board]);
      }
      if (url.endsWith("/api/boards/1")) {
        return response({
          ...board,
          messages: [
            { id: 1, role: "user", content: "image of a fox", citations: [], source: "user", created_at: 1, media: [] },
            { id: 2, role: "assistant", content: "The image-generation rate limit was reached.", citations: [], source: "unavailable", created_at: 2, media: [], media_error: { code: "rate_limit", retryable: true } },
          ],
        });
      }
      if (url.endsWith("/api/chat/stream")) {
        return response({ answer: "Retrying", citations: [], source: "general", media: [] });
      }
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Retry generation" }));

    const chatCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/chat/stream"));
    expect(JSON.parse(String(chatCall?.[1]?.body)).question).toBe("image of a fox");
  });

  it("does not offer retry for permanent media failures", async () => {
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) return response([board]);
      if (url.endsWith("/api/boards/1")) return response({
        ...board,
        messages: [{ id: 2, role: "assistant", content: "Image generation requires credits.", citations: [], source: "unavailable", created_at: 2, media: [], media_error: { code: "insufficient_quota", retryable: false } }],
      });
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    expect(await screen.findByText("Image generation requires credits.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry generation" })).not.toBeInTheDocument();
  });

  it("includes the bearer token and board id in chat requests", async () => {
    const fetchMock = mockBoardApi();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await userEvent.click(await screen.findByText("How do I invite a teammate?"));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/api/chat/stream"))).toBe(true));
    const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/chat/stream"));
    expect((call?.[1]?.headers as Record<string, string>).Authorization).toBe("Bearer test-token");
    expect(JSON.parse(String(call?.[1]?.body)).board_id).toBe(1);
  });

  it("attaches a text file to a question", async () => {
    const fetchMock = mockBoardApi();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    const file = new File(["Refunds take five days."], "notes.txt", { type: "text/plain" });
    await userEvent.upload(await screen.findByLabelText("Attach files"), file);
    expect(await screen.findByText("notes.txt", { exact: false })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Ask FlowDesk Support"), "What does this say?");
    await userEvent.click(screen.getByLabelText("Send question"));
    const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/chat/stream"));
    expect(JSON.parse(String(call?.[1]?.body)).attachment_ids).toEqual(["upload-notes.txt"]);
  });

  it("keeps the typing indicator on the board that sent the question", async () => {
    let resolveChat!: (value: unknown) => void;
    const pendingChat = new Promise((resolve) => { resolveChat = resolve; });
    const second = { ...board, id: 2, title: "Second chat" };
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) {
        return response([board, second]);
      }
      if (url.endsWith("/api/boards/1")) return response({ ...board, messages: [] });
      if (url.endsWith("/api/boards/2")) return response({ ...second, messages: [] });
      if (url.endsWith("/api/chat/stream")) return pendingChat;
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<App />);
    await userEvent.click(await screen.findByText("How do I invite a teammate?"));
    await waitFor(() => expect(container.querySelector(".typing")).toBeInTheDocument());
    await userEvent.click(screen.getByText("Second chat"));
    await waitFor(() => expect(container.querySelector(".typing")).not.toBeInTheDocument());
    expect(screen.getByLabelText("Ask FlowDesk Support")).toBeEnabled();
    resolveChat({ ok: true, json: async () => ({ answer: "Done", citations: [], source: "general" }) });
  });

  it("stops the active chat request", async () => {
    const fetchMock = vi.fn((url: string, options?: RequestInit) => {
      if (url.endsWith("/api/boards") && (!options?.method || options.method === "GET")) return response([board]);
      if (url.endsWith("/api/boards/1")) return response({ ...board, messages: [] });
      if (url.endsWith("/api/chat/stream")) return new Promise((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      });
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<App />);
    await userEvent.click(await screen.findByText("How do I invite a teammate?"));
    await userEvent.click(await screen.findByLabelText("Stop generating"));
    await waitFor(() => expect(container.querySelector(".typing")).not.toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("authentication", () => {
  it("shows the sign-in screen without a session", () => {
    sessionStorage.clear();
    render(<App />);
    expect(screen.getByRole("heading", { name: "Sign in to your account" })).toBeInTheDocument();
  });

  it("shows and hides the password without clearing it", async () => {
    sessionStorage.clear();
    render(<App />);
    const password = screen.getByLabelText("Password");
    await userEvent.type(password, "secret-123");
    expect(password).toHaveAttribute("type", "password");
    await userEvent.click(screen.getByLabelText("Show password"));
    expect(password).toHaveAttribute("type", "text");
    expect(password).toHaveValue("secret-123");
    await userEvent.click(screen.getByLabelText("Hide password"));
    expect(password).toHaveAttribute("type", "password");
  });
});
