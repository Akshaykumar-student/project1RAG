import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  authenticate, BoardDetail, BoardSummary, ChatMedia, Citation, createBoard,
  deleteBoard, deleteTemporaryUpload, getAttachmentBlob, getAttachmentPreview, getBoard, getMediaBlob, listBoards, renameBoard,
  requestPasswordReset, resetPassword,
  searchMessages, SearchResult, StoredAttachment, StoredMessage, streamFlowDesk, TemporaryUpload,
  uploadAttachment, User,
} from "./api";

type SpeechRecognitionEventLike = {
  results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }>;
};

type SpeechRecognitionErrorLike = { error: string };

type BrowserSpeechRecognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecognitionConstructor = new () => BrowserSpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

const samples = ["How do I invite a teammate?", "What happens when I cancel my subscription?", "How can I fix error FD-403?"];
const messageTimeFormatter = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
const MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024;
const MAX_TOTAL_ATTACHMENT_BYTES = 100 * 1024 * 1024;

function readBlobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error("Could not read the selected text file."));
    reader.readAsText(blob);
  });
}
const supportedAttachmentTypes: Record<string, string> = {
  txt: "text/plain", md: "text/markdown", csv: "text/csv", json: "application/json",
  pdf: "application/pdf", jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png",
  webp: "image/webp",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
};

type SelectedAttachment = TemporaryUpload & {
  clientId: string;
  progress: number;
  uploading: boolean;
  localUrl: string;
  localText?: string;
};

function formatFileSize(size: number) {
  return size < 1024 * 1024 ? `${Math.ceil(size / 1024)} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function SparkIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l1.7 5.3L19 9l-5.3 1.7L12 16l-1.7-5.3L5 9l5.3-1.7L12 2Z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" /></svg>;
}

function EyeIcon({ hidden }: { hidden: boolean }) {
  return <svg className="eye-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
    <circle cx="12" cy="12" r="2.6" />
    {hidden && <path d="m4 4 16 16" />}
  </svg>;
}

function CopyIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="8" y="8" width="11" height="11" rx="2" /><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" /></svg>;
}

function MicrophoneIcon() {
  return <svg className="microphone-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M9 21h6" /></svg>;
}

export default function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem("flowdesk_token") ?? "");
  const [user, setUser] = useState<User | null>(() => {
    const saved = sessionStorage.getItem("flowdesk_user");
    return saved ? JSON.parse(saved) : null;
  });
  const [boards, setBoards] = useState<BoardSummary[]>([]);
  const [activeBoard, setActiveBoard] = useState<BoardDetail | null>(null);
  const [question, setQuestion] = useState("");
  const [attachments, setAttachments] = useState<SelectedAttachment[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [recentsOpen, setRecentsOpen] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [highlightedMessageId, setHighlightedMessageId] = useState<number | null>(null);
  const [loadingBoardIds, setLoadingBoardIds] = useState<Set<number>>(() => new Set());
  const [streamingBoardIds, setStreamingBoardIds] = useState<Set<number>>(() => new Set());
  const [attachmentAnalysisBoardIds, setAttachmentAnalysisBoardIds] = useState<Set<number>>(() => new Set());
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState("");
  const conversationRef = useRef<HTMLDivElement>(null);
  const renderedBoardIdRef = useRef<number | null>(null);
  const requestControllersRef = useRef<Map<number, AbortController>>(new Map());
  const searchInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const speechBaseTextRef = useRef("");
  const activeBoardLoading = activeBoard ? loadingBoardIds.has(activeBoard.id) : false;
  const activeBoardStreaming = activeBoard ? streamingBoardIds.has(activeBoard.id) : false;
  const activeAttachmentAnalysis = activeBoard ? attachmentAnalysisBoardIds.has(activeBoard.id) : false;
  const speechSupported = typeof window !== "undefined" && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

  const stopSpeechRecognition = useCallback((abort = false) => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    recognitionRef.current = null;
    recognition.onresult = null;
    recognition.onerror = null;
    recognition.onend = null;
    try {
      if (abort) recognition.abort();
      else recognition.stop();
    } catch {
      // The browser may already have ended recognition.
    }
    setIsListening(false);
  }, []);

  useEffect(() => {
    if (token && user) void loadInitialBoards(token);
  }, [token, user]);

  useEffect(() => {
    const conversation = conversationRef.current;
    const boardId = activeBoard?.id;
    if (!conversation || !boardId) return;
    if (renderedBoardIdRef.current !== boardId) {
      renderedBoardIdRef.current = boardId;
      conversation.scrollTop = conversation.scrollHeight;
      return;
    }
    conversation.scrollTo?.({ top: conversation.scrollHeight, behavior: "smooth" });
  }, [activeBoard?.id, activeBoard?.messages.length, activeBoardLoading]);

  useEffect(() => {
    const query = searchQuery.trim();
    if (query.length < 2) {
      setSearchResults([]);
      setSearchLoading(false);
      setSearchError("");
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setSearchLoading(true);
      setSearchError("");
      void searchMessages(token, query, controller.signal)
        .then(setSearchResults)
        .catch((caught) => {
          if (!controller.signal.aborted) {
            setSearchError(caught instanceof Error ? caught.message : "Could not search chats.");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setSearchLoading(false);
        });
    }, 250);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [searchQuery, token]);

  useEffect(() => {
    if (highlightedMessageId === null) return;
    const message = document.getElementById(`message-${highlightedMessageId}`);
    if (!message) return;
    message.scrollIntoView({ behavior: "smooth", block: "center" });
    const timer = window.setTimeout(() => setHighlightedMessageId(null), 2200);
    return () => window.clearTimeout(timer);
  }, [activeBoard?.id, highlightedMessageId]);

  useEffect(() => {
    stopSpeechRecognition(true);
  }, [activeBoard?.id, stopSpeechRecognition]);

  useEffect(() => () => stopSpeechRecognition(true), [stopSpeechRecognition]);

  function toggleSpeechRecognition() {
    if (isListening) {
      stopSpeechRecognition();
      return;
    }
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      setError("Voice input isn't supported by this browser. Try Chrome or Edge.");
      return;
    }

    const recognition = new Recognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = navigator.language || "en-US";
    speechBaseTextRef.current = question.trim();
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results, (result) => result[0]?.transcript ?? "").join(" ").trim();
      setQuestion([speechBaseTextRef.current, transcript].filter(Boolean).join(" "));
      if (Array.from(event.results).some((result) => result.isFinal)) recognition.stop();
    };
    recognition.onerror = (event) => {
      const messages: Record<string, string> = {
        "not-allowed": "Microphone permission was denied. Allow microphone access and try again.",
        "audio-capture": "No working microphone was found.",
        "no-speech": "I couldn't hear any speech. Please try again.",
        network: "Voice recognition couldn't connect. Please check your connection and try again.",
      };
      setError(messages[event.error] ?? "Voice recognition failed. Please try again.");
    };
    recognition.onend = () => {
      if (recognitionRef.current === recognition) recognitionRef.current = null;
      setIsListening(false);
    };
    recognitionRef.current = recognition;
    setError("");
    try {
      recognition.start();
      setIsListening(true);
    } catch {
      recognitionRef.current = null;
      setIsListening(false);
      setError("Voice recognition couldn't start. Please try again.");
    }
  }

  async function loadInitialBoards(accessToken: string) {
    try {
      let items = await listBoards(accessToken);
      if (!items.length) items = [await createBoard(accessToken)];
      setBoards(items);
      setActiveBoard(await getBoard(accessToken, items[0].id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load chats.");
    }
  }

  async function selectBoard(boardId: number) {
    setHighlightedMessageId(null);
    setError("");
    try { setActiveBoard(await getBoard(token, boardId)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Could not open this chat."); }
  }

  async function openSearchResult(result: SearchResult) {
    setError("");
    try {
      const board = await getBoard(token, result.board_id);
      setActiveBoard(board);
      setHighlightedMessageId(result.message_id);
    } catch (caught) {
      setSearchError(caught instanceof Error ? caught.message : "Could not open this result.");
    }
  }

  function openSearch() {
    setSidebarCollapsed(false);
    window.requestAnimationFrame(() => searchInputRef.current?.focus());
  }

  function toggleRecents() {
    if (searchQuery.trim()) {
      setSearchQuery("");
      setRecentsOpen(true);
      return;
    }
    setRecentsOpen((current) => !current);
  }

  async function newBoard() {
    const board = await createBoard(token);
    setBoards((current) => [board, ...current]);
    setActiveBoard({ ...board, messages: [] });
  }

  async function editBoard(board: BoardSummary) {
    const title = window.prompt("Rename chat", board.title)?.trim();
    if (!title || title === board.title) return;
    const updated = await renameBoard(token, board.id, title);
    setBoards((current) => current.map((item) => item.id === board.id
      ? { ...item, title: updated.title, updated_at: updated.updated_at } : item));
    if (activeBoard?.id === board.id) setActiveBoard(updated);
  }

  async function removeBoard(boardId: number) {
    if (!window.confirm("Delete this chat and all of its messages?")) return;
    await deleteBoard(token, boardId);
    const remaining = boards.filter((board) => board.id !== boardId);
    if (remaining.length) {
      setBoards(remaining);
      setActiveBoard(await getBoard(token, remaining[0].id));
    } else {
      const created = await createBoard(token);
      setBoards([created]);
      setActiveBoard({ ...created, messages: [] });
    }
  }

  async function submit(value: string) {
    const cleaned = value.trim();
    if (!cleaned || !activeBoard || loadingBoardIds.has(activeBoard.id)
      || attachments.some((attachment) => attachment.uploading)) return;
    stopSpeechRecognition(true);
    const boardId = activeBoard.id;
    const submittedAttachments = attachments;
    const optimisticAttachments: StoredAttachment[] = submittedAttachments.map((attachment, index) => ({
      id: -(Date.now() + index),
      filename: attachment.filename,
      mime_type: attachment.mime_type,
      size: attachment.size,
      preview_type: attachment.mime_type === "application/pdf" ? "pdf"
        : attachment.mime_type.startsWith("image/") ? "image"
          : ["text/plain", "text/markdown", "text/csv", "application/json"].includes(attachment.mime_type) ? "text" : "document",
      local_content: attachment.localText,
      local_url: attachment.localUrl,
    }));
    const optimistic: StoredMessage = { id: -Date.now(), role: "user", content: cleaned, citations: [], source: "user", created_at: Math.floor(Date.now() / 1000), media: [], attachments: optimisticAttachments };
    const streamingMessageId = optimistic.id - 1;
    let streamedText = "";
    setActiveBoard((current) => current && ({ ...current, messages: [...current.messages, optimistic] }));
    setQuestion(""); setAttachments([]); setError("");
    setLoadingBoardIds((current) => new Set(current).add(boardId));
    if (submittedAttachments.length) {
      setAttachmentAnalysisBoardIds((current) => new Set(current).add(boardId));
    }
    const controller = new AbortController();
    requestControllersRef.current.set(boardId, controller);
    try {
      await streamFlowDesk(
        cleaned,
        token,
        boardId,
        submittedAttachments.map((attachment) => attachment.id),
        controller.signal,
        (delta) => {
          streamedText += delta;
          setStreamingBoardIds((current) => new Set(current).add(boardId));
          setActiveBoard((current) => {
            if (current?.id !== boardId) return current;
            const existing = current.messages.findIndex((message) => message.id === streamingMessageId);
            const streamingMessage: StoredMessage = {
              id: streamingMessageId,
              role: "assistant",
              content: streamedText,
              citations: [],
              source: "general",
              created_at: Math.floor(Date.now() / 1000),
              media: [],
              attachments: [],
            };
            return {
              ...current,
              messages: existing >= 0
                ? current.messages.map((message) => message.id === streamingMessageId ? streamingMessage : message)
                : [...current.messages, streamingMessage],
            };
          });
        },
      );
      const [detail, summaries] = await Promise.all([getBoard(token, boardId), listBoards(token)]);
      setActiveBoard((current) => current?.id === boardId ? detail : current);
      setBoards(summaries);
      submittedAttachments.forEach((attachment) => URL.revokeObjectURL(attachment.localUrl));
    } catch (caught) {
      if (controller.signal.aborted) {
        setActiveBoard((current) => {
          if (current?.id !== boardId) return current;
          if (streamedText) {
            return {
              ...current,
              messages: current.messages.map((message) => message.id === streamingMessageId
                ? { ...message, content: `${message.content}\n\n[Response stopped]` }
                : message),
            };
          }
          return { ...current, messages: current.messages.filter((message) => message.id !== optimistic.id) };
        });
      } else {
        setError(caught instanceof Error ? caught.message : "Something went wrong.");
        setActiveBoard((current) => current?.id === boardId
          ? {
            ...current,
            messages: streamedText
              ? current.messages.map((message) => message.id === streamingMessageId
                ? { ...message, content: `${message.content}\n\n[Response interrupted]` }
                : message)
              : current.messages.filter((message) => message.id !== optimistic.id),
          }
          : current);
        if (!streamedText && renderedBoardIdRef.current === boardId) {
          setQuestion(cleaned);
          setAttachments(submittedAttachments);
        }
      }
    } finally {
      if (requestControllersRef.current.get(boardId) === controller) {
        requestControllersRef.current.delete(boardId);
      }
      setLoadingBoardIds((current) => {
        const next = new Set(current);
        next.delete(boardId);
        return next;
      });
      setAttachmentAnalysisBoardIds((current) => {
        const next = new Set(current);
        next.delete(boardId);
        return next;
      });
      setStreamingBoardIds((current) => {
        const next = new Set(current);
        next.delete(boardId);
        return next;
      });
    }
  }

  function stopActiveResponse() {
    if (activeBoard) requestControllersRef.current.get(activeBoard.id)?.abort();
  }

  function retryMedia(messageId: number) {
    if (!activeBoard) return;
    const messageIndex = activeBoard.messages.findIndex((message) => message.id === messageId);
    for (let index = messageIndex - 1; index >= 0; index -= 1) {
      const message = activeBoard.messages[index];
      if (message.role === "user") {
        void submit(message.content.split("\n\nAttached:")[0]);
        return;
      }
    }
  }

  async function attachFiles(files: FileList | null) {
    if (!files) return;
    const available = Math.max(0, 3 - attachments.length);
    const selected = Array.from(files).slice(0, available);
    try {
      const selectedSize = selected.reduce((sum, file) => sum + file.size, 0);
      const totalSize = attachments.reduce((sum, item) => sum + item.size, 0) + selectedSize;
      if (totalSize > MAX_TOTAL_ATTACHMENT_BYTES) {
        throw new Error("Attachments cannot exceed 100 MB combined.");
      }
      const pending = await Promise.all(selected.map(async (file) => {
        const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
        const mimeType = supportedAttachmentTypes[extension];
        if (!mimeType) throw new Error(`${file.name} has an unsupported file format.`);
        if (!file.size) throw new Error(`${file.name} is empty.`);
        if (file.size > MAX_ATTACHMENT_BYTES) throw new Error(`${file.name} is larger than 100 MB.`);
        const previewType: SelectedAttachment["preview_type"] = mimeType === "application/pdf" ? "pdf"
          : mimeType.startsWith("image/") ? "image"
            : ["text/plain", "text/markdown", "text/csv", "application/json"].includes(mimeType) ? "text" : "document";
        return {
          id: "",
          clientId: crypto.randomUUID(),
          filename: file.name,
          mime_type: mimeType,
          size: file.size,
          preview_type: previewType,
          expires_at: 0,
          progress: 0,
          uploading: true,
          localUrl: URL.createObjectURL(file),
          localText: previewType === "text"
            ? await readBlobText(file.slice(0, 50_001))
            : undefined,
          file,
        };
      }));
      setAttachments((current) => [...current, ...pending]);
      setError("");
      await Promise.all(pending.map(async (item) => {
        try {
          const uploaded = await uploadAttachment(token, item.file, (progress) => {
            setAttachments((current) => current.map((attachment) =>
              attachment.clientId === item.clientId ? { ...attachment, progress } : attachment));
          });
          setAttachments((current) => current.map((attachment) =>
            attachment.clientId === item.clientId
              ? { ...attachment, ...uploaded, progress: 100, uploading: false }
              : attachment));
        } catch (caught) {
          URL.revokeObjectURL(item.localUrl);
          setAttachments((current) => current.filter(
            (attachment) => attachment.clientId !== item.clientId));
          throw caught;
        }
      }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not attach this file.");
    }
  }

  function handleAuthenticated(accessToken: string, authenticatedUser: User) {
    sessionStorage.setItem("flowdesk_token", accessToken);
    sessionStorage.setItem("flowdesk_user", JSON.stringify(authenticatedUser));
    setToken(accessToken); setUser(authenticatedUser);
  }

  function logout() {
    stopSpeechRecognition(true);
    requestControllersRef.current.forEach((controller) => controller.abort());
    requestControllersRef.current.clear();
    sessionStorage.clear(); setToken(""); setUser(null); setBoards([]); setActiveBoard(null);
  }

  if (!token || !user) return <AuthScreen onAuthenticated={handleAuthenticated} />;

  return <main className={`app-shell boards-layout ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
    <aside className="sidebar boards-sidebar">
      <div className="brand"><span className="brand-mark"><SparkIcon /></span><span className="sidebar-label">FlowDesk</span></div>
      <button className="sidebar-toggle" type="button" onClick={() => setSidebarCollapsed((current) => !current)} aria-expanded={!sidebarCollapsed} aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"} title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>{sidebarCollapsed ? "›" : "‹"}</button>
      {sidebarCollapsed
        ? <button className="search-rail-button" type="button" onClick={openSearch} aria-label="Search chats" title="Search chats">⌕</button>
        : <div className="chat-search"><span aria-hidden="true">⌕</span><input ref={searchInputRef} type="search" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search chats…" aria-label="Search chats" maxLength={100} />{searchQuery && <button type="button" onClick={() => setSearchQuery("")} aria-label="Clear search">×</button>}</div>}
      <button className="new-chat" type="button" onClick={() => void newBoard()} title="New chat"><span>＋</span><span className="sidebar-label">New chat</span></button>
      <button className="recents-toggle" type="button" onClick={toggleRecents} aria-expanded={searchQuery.trim().length >= 2 ? false : recentsOpen}><span>Recents</span><span aria-hidden="true">{searchQuery.trim().length >= 2 || !recentsOpen ? "⌄" : "⌃"}</span></button>
      {searchQuery.trim().length >= 2 ? <div className="search-results" aria-live="polite">{searchLoading && <p className="search-status">Searching…</p>}{searchError && <p className="search-status search-error" role="alert">{searchError}</p>}{!searchLoading && !searchError && !searchResults.length && <p className="search-status">No matching messages.</p>}{searchResults.map((result) => <button type="button" key={result.message_id} onClick={() => void openSearchResult(result)}><strong>{result.board_title}</strong><small>{result.role === "user" ? "Your question" : "FlowDesk answer"}</small><span>{result.snippet}</span></button>)}</div> : recentsOpen ? <nav className="board-list" aria-label="Recent chats">{boards.map((board) =>
        <div className={`board-item ${activeBoard?.id === board.id ? "active" : ""}`} key={board.id}>
          <button className="board-open" type="button" onClick={() => void selectBoard(board.id)}><span>{board.title}</span><small>{board.message_count} messages</small></button>
          <div className="board-actions"><button type="button" aria-label={`Rename ${board.title}`} onClick={() => void editBoard(board)}>✎</button><button type="button" aria-label={`Delete ${board.title}`} onClick={() => void removeBoard(board.id)}>×</button></div>
        </div>)}</nav> : <div className="sidebar-space" />}
      <div className="sidebar-user"><strong>{user.name}</strong><small>{user.email}</small></div>
    </aside>
    <section className="chat-panel">
      <header className="chat-header"><div><p className="eyebrow">FLOWDESK SUPPORT</p><h2>{activeBoard?.title ?? "Loading chat…"}</h2></div><div className="user-menu"><button type="button" onClick={logout}>Sign out</button></div></header>
      <div className="conversation" ref={conversationRef} aria-live="polite" aria-label="Conversation messages">
        {!activeBoard?.messages.length && <article className="message assistant"><div className="avatar"><SparkIcon /></div><div className="message-content"><p>Start this chat by asking a question about FlowDesk.</p></div></article>}
        {activeBoard?.messages.map((message) => <MessageBubble key={message.id} message={message} highlighted={message.id === highlightedMessageId} token={token} onRetry={message.media_error?.retryable ? () => retryMedia(message.id) : undefined} />)}
        {activeBoardLoading && !activeBoardStreaming && <article className="message assistant"><div className="avatar"><SparkIcon /></div><div className="typing"><span /><span /><span /></div></article>}
        {error && <div className="error" role="alert">{error}</div>}
        <div className="conversation-end" aria-hidden="true" />
      </div>
      {!activeBoard?.messages.length && <div className="suggestions"><p>Try asking</p><div>{samples.map((sample) => <button key={sample} type="button" onClick={() => void submit(sample)}>{sample}</button>)}</div></div>}
      <div className="composer-area">
        {activeAttachmentAnalysis && <p className="attachment-analysis" role="status">Uploading and analyzing attachments…</p>}
        {!!attachments.length && <div className="attachment-list">{attachments.map((attachment, index) => <span key={attachment.clientId}>▤ <span>{attachment.filename}<small>{attachment.filename.split(".").pop()?.toUpperCase()} · {formatFileSize(attachment.size)}{attachment.uploading ? ` · Uploading ${attachment.progress}%` : ""}</small></span><button type="button" aria-label={`Remove ${attachment.filename}`} onClick={() => { URL.revokeObjectURL(attachment.localUrl); if (attachment.id) void deleteTemporaryUpload(token, attachment.id).catch(() => undefined); setAttachments((current) => current.filter((_, itemIndex) => itemIndex !== index)); }}>×</button></span>)}</div>}
        <form className="composer" onSubmit={(event: FormEvent) => { event.preventDefault(); void submit(question); }}>
          <label className="attach-button" title="Attach files">＋<span className="sr-only">Attach files</span><input aria-label="Attach files" type="file" accept=".txt,.md,.csv,.json,.pdf,.docx,.xlsx,.pptx,.jpg,.jpeg,.png,.webp" multiple onChange={(event) => { void attachFiles(event.target.files); event.target.value = ""; }} disabled={activeBoardLoading || attachments.length >= 3} /></label>
          <label htmlFor="question" className="sr-only">Ask FlowDesk Support</label><input id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about FlowDesk…" maxLength={1000} disabled={activeBoardLoading || !activeBoard} />
          <button className={`microphone-button ${isListening ? "listening" : ""}`} type="button" onClick={toggleSpeechRecognition} disabled={!activeBoard || activeBoardLoading} aria-label={isListening ? "Stop listening" : "Start voice input"} aria-pressed={isListening} title={speechSupported ? (isListening ? "Stop listening" : "Start voice input") : "Voice input is not supported by this browser"}><MicrophoneIcon /></button>
          {activeBoardLoading
            ? <button className="stop-button" type="button" onClick={stopActiveResponse} aria-label="Stop generating"><span /></button>
            : <button className="send-button" type="submit" disabled={!activeBoard || question.trim().length < 2 || attachments.some((attachment) => attachment.uploading)} aria-label="Send question">↑</button>}
        </form>
      </div>
      <footer>Messages are saved to this board automatically.</footer>
    </section>
  </main>;
}

function MessageBubble({ message, highlighted, token, onRetry }: { message: StoredMessage; highlighted: boolean; token: string; onRetry?: () => void }) {
  const [copied, setCopied] = useState(false);
  const sourceLabel = message.source === "documentation" ? "FlowDesk documentation" : message.source === "web" ? "Web sources" : "General knowledge";
  const createdAt = new Date(message.created_at * 1000);

  async function copyMessage() {
    try {
      await navigator.clipboard.writeText(message.content.split("\n\nAttached:")[0]);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  const visibleContent = message.content.split("\n\nAttached:")[0];
  return <article id={`message-${message.id}`} className={`message ${message.role} ${highlighted ? "message-highlighted" : ""}`}>{message.role === "assistant" && <div className="avatar"><SparkIcon /></div>}<div className="message-content">{!!message.attachments?.length && <div className="message-attachments">{message.attachments.map((attachment) => <AttachmentCard key={attachment.id} attachment={attachment} token={token} />)}</div>}<p>{visibleContent}</p>{(message.media ?? []).map((media) => <InlineMedia key={media.id} media={media} token={token} />)}{message.role === "assistant" && message.source !== "unavailable" && !(message.media ?? []).length && <span className={`answer-source ${message.source === "documentation" ? "grounded" : message.source === "web" ? "web" : "general"}`}>{sourceLabel}</span>}<Citations citations={(message.media ?? []).length ? [] : message.citations} />{onRetry && <button className="retry-media" type="button" onClick={onRetry}>Retry generation</button>}<div className="message-actions"><button className="copy-message" type="button" onClick={() => void copyMessage()} aria-label={copied ? "Message copied" : `Copy ${message.role === "user" ? "question" : "answer"}`} title={copied ? "Copied" : "Copy text"}><CopyIcon /><span>{copied ? "Copied" : "Copy"}</span></button><time dateTime={createdAt.toISOString()} title={createdAt.toLocaleString()}>{messageTimeFormatter.format(createdAt)}</time></div></div></article>;
}

function AttachmentCard({ attachment, token }: { attachment: StoredAttachment; token: string }) {
  const [open, setOpen] = useState(false);
  const closePreview = useCallback(() => setOpen(false), []);
  const extension = attachment.filename.split(".").pop()?.toUpperCase() ?? "FILE";
  return <><button className="message-attachment-card" type="button" onClick={() => setOpen(true)} aria-label={`Open ${attachment.filename}`}><span className={`file-icon ${attachment.preview_type}`}>{extension === "PDF" ? "PDF" : extension.slice(0, 4)}</span><span><strong>{attachment.filename}</strong><small>{extension} · {formatFileSize(attachment.size)}{attachment.id < 0 ? " · Analyzing" : ""}</small></span></button>{open && <AttachmentPreview attachment={attachment} token={token} onClose={closePreview} />}</>;
}

function AttachmentPreview({ attachment, token, onClose }: { attachment: StoredAttachment; token: string; onClose: () => void }) {
  const [source, setSource] = useState("");
  const [text, setText] = useState("");
  const [previewError, setPreviewError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = "";
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", closeOnEscape);
    if (attachment.id < 0 && attachment.local_url
      && (attachment.preview_type === "pdf" || attachment.preview_type === "image")) {
      setSource(attachment.local_url);
    } else if (attachment.id < 0 && attachment.local_content) {
      if (attachment.local_encoding === "utf8" || attachment.preview_type === "text") {
        setText(attachment.local_content);
      } else if (attachment.preview_type === "pdf" || attachment.preview_type === "image") {
        try {
          const decoded = window.atob(attachment.local_content);
          const bytes = Uint8Array.from(decoded, (character) => character.charCodeAt(0));
          objectUrl = URL.createObjectURL(new Blob([bytes], { type: attachment.mime_type }));
          setSource(objectUrl);
        } catch {
          setPreviewError("This attachment preview could not be prepared.");
        }
      } else {
        setText("This file is being analyzed. Its extracted preview will be available when the response is ready.");
      }
    } else if (attachment.preview_type === "pdf" || attachment.preview_type === "image") {
      void getAttachmentBlob(token, attachment.id, controller.signal).then((blob) => {
        objectUrl = URL.createObjectURL(blob); setSource(objectUrl);
      }).catch((caught) => { if (!controller.signal.aborted) setPreviewError(caught instanceof Error ? caught.message : "Could not open this attachment."); });
    } else {
      void getAttachmentPreview(token, attachment.id, controller.signal).then((result) => setText(result.content)).catch((caught) => { if (!controller.signal.aborted) setPreviewError(caught instanceof Error ? caught.message : "Could not open this attachment."); });
    }
    return () => { controller.abort(); window.removeEventListener("keydown", closeOnEscape); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [attachment, onClose, token]);
  return <div className="attachment-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section className="attachment-modal" role="dialog" aria-modal="true" aria-label={`Preview ${attachment.filename}`}><header><div><strong>{attachment.filename}</strong><small>{attachment.filename.split(".").pop()?.toUpperCase()} · {formatFileSize(attachment.size)}</small></div><button type="button" onClick={onClose} aria-label="Close attachment preview">×</button></header><div className="attachment-preview-body">{previewError ? <div className="error" role="alert">{previewError}</div> : attachment.preview_type === "image" && source ? <img src={source} alt={attachment.filename} /> : attachment.preview_type === "pdf" && source ? <iframe src={source} title={attachment.filename} /> : text ? <pre>{text}</pre> : <p className="attachment-preview-loading">Loading preview…</p>}</div></section></div>;
}

function InlineMedia({ media, token }: { media: ChatMedia; token: string }) {
  const [source, setSource] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = "";
    setSource("");
    setFailed(false);
    void getMediaBlob(token, media.id, controller.signal)
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      })
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [media.id, token]);

  if (failed) return <div className="generated-media-status" role="status">Could not load this generated media.</div>;
  if (!source) return <div className="generated-media-status" role="status">Loading generated media…</div>;
  return <div className="generated-media">{media.type === "video" ? <video src={source} controls playsInline preload="metadata" aria-label={media.alt} /> : <img src={source} alt={media.alt} />}</div>;
}

function Citations({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return <div className="sources"><span className="sources-label">Sources</span>{citations.map((citation) => citation.source_type === "web" && citation.url ? <a className="source-card" key={citation.url} href={citation.url} target="_blank" rel="noreferrer"><span className="document-icon">↗</span><span><strong>{citation.label}</strong><small>{new URL(citation.url).hostname}</small></span></a> : <span className="source-card" key={citation.filename}><span className="document-icon">▤</span><span><strong>{citation.label}</strong><small>{citation.filename}</small></span></span>)}</div>;
}

function AuthScreen({ onAuthenticated }: { onAuthenticated: (token: string, user: User) => void }) {
  type AuthMode = "login" | "register" | "forgot" | "reset";
  const resetToken = new URLSearchParams(window.location.search).get("reset_token") ?? "";
  const [mode, setMode] = useState<AuthMode>(resetToken ? "reset" : "login");
  const [name, setName] = useState(""); const [email, setEmail] = useState(""); const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false); const [error, setError] = useState(""); const [message, setMessage] = useState("");
  const passwordRequirements = [
    [password.length >= 10, "10 or more characters"],
    [/[a-z]/.test(password), "One lowercase letter"],
    [/[A-Z]/.test(password), "One uppercase letter"],
    [/\d/.test(password), "One number"],
    [/[^A-Za-z0-9]/.test(password), "One special character"],
  ] as const;
  async function submitAuth(event: FormEvent) {
    event.preventDefault(); setLoading(true); setError(""); setMessage("");
    try {
      if (mode === "forgot") {
        setMessage(await requestPasswordReset(email));
      } else if (mode === "reset") {
        if (password !== confirmPassword) throw new Error("Passwords do not match.");
        setMessage(await resetPassword(resetToken, password));
        window.history.replaceState({}, "", window.location.pathname);
        setPassword(""); setConfirmPassword(""); setMode("login");
      } else {
        const result = await authenticate(mode, email, password, name);
        onAuthenticated(result.access_token, result.user);
      }
    }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Authentication failed."); }
    finally { setLoading(false); }
  }
  const title = mode === "login" ? "Sign in to your account" : mode === "register" ? "Create your account" : mode === "forgot" ? "Reset your password" : "Choose a new password";
  const submitLabel = mode === "login" ? "Sign in" : mode === "register" ? "Create account" : mode === "forgot" ? "Send reset link" : "Reset password";
  const switchMode = (next: AuthMode) => { setMode(next); setShowPassword(false); setError(""); setMessage(""); };
  return <main className="auth-shell"><section className="auth-brand"><div className="brand"><span className="brand-mark"><SparkIcon /></span><span>FlowDesk</span></div><div><p className="eyebrow">AI-POWERED SUPPORT</p><h1>Your answers are already here.</h1><p>Sign in to search trusted documentation and keep conversations organized.</p></div></section><section className="auth-panel"><form className="auth-card" onSubmit={submitAuth}><p className="eyebrow">WELCOME TO FLOWDESK</p><h2>{title}</h2><p className="auth-subtitle">{mode === "forgot" ? "Enter your account email and we'll send a secure reset link." : mode === "reset" ? "Your link is verified when you save the new password." : "Continue to your saved support chats."}</p>{mode === "register" && <label>Full name<input value={name} onChange={(event) => setName(event.target.value)} minLength={2} required /></label>}{mode !== "reset" && <label>Email address<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>}{mode !== "forgot" && <label htmlFor="auth-password">{mode === "reset" ? "New password" : "Password"}<div className="password-field"><input id="auth-password" type={showPassword ? "text" : "password"} value={password} onChange={(event) => setPassword(event.target.value)} minLength={mode === "login" ? 1 : 10} required /><button type="button" onClick={() => setShowPassword((current) => !current)} aria-label={showPassword ? "Hide password" : "Show password"} title={showPassword ? "Hide password" : "Show password"}><EyeIcon hidden={!showPassword} /></button></div></label>}{(mode === "register" || mode === "reset") && <ul className="password-requirements" aria-label="Password requirements">{passwordRequirements.map(([valid, requirement]) => <li className={valid ? "valid" : ""} key={requirement}>{valid ? "✓" : "○"} {requirement}</li>)}</ul>}{mode === "reset" && <label>Confirm new password<input type={showPassword ? "text" : "password"} value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} minLength={10} required /></label>}{error && <div className="error" role="alert">{error}</div>}{message && <div className="auth-message" role="status">{message}</div>}<button className="auth-submit" type="submit" disabled={loading}>{loading ? "Please wait…" : submitLabel}</button>{mode === "login" && <p className="auth-switch"><button type="button" onClick={() => switchMode("forgot")}>Forgot password?</button></p>}<p className="auth-switch">{mode === "login" ? "New to FlowDesk?" : "Already have an account?"} <button type="button" onClick={() => switchMode(mode === "login" ? "register" : "login")}>{mode === "login" ? "Create an account" : "Sign in"}</button></p></form></section></main>;
}
