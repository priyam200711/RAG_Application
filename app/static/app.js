// Client-side Controller for Enterprise RAG UI
document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const queryInput = document.getElementById("query-input");
  const sendButton = document.getElementById("send-button");
  const streamToggle = document.getElementById("stream-toggle");

  const btnSeedSamples = document.getElementById("btn-seed-samples");
  const btnOpenIngest = document.getElementById("btn-open-ingest");
  const btnClearIndex = document.getElementById("btn-clear-index");

  const ingestModal = document.getElementById("ingest-modal");
  const modalClose = document.getElementById("modal-close");
  const btnCancelIngest = document.getElementById("btn-cancel-ingest");
  const ingestForm = document.getElementById("ingest-form");

  const valLlmProvider = document.getElementById("val-llm-provider");
  const valVectordb = document.getElementById("val-vectordb");
  const valChunksCount = document.getElementById("val-chunks-count");
  const valEmbedding = document.getElementById("val-embedding");
  const systemStatusText = document.getElementById("system-status-text");
  const sourceList = document.getElementById("source-list");
  const sourceCount = document.getElementById("source-count");

  // Keep chat history in memory for conversational RAG
  const chatHistory = [];

  // 1. Fetch System Health & Stats
  async function refreshSystemStatus() {
    try {
      const res = await fetch("/api/v1/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      valLlmProvider.textContent = `${data.llm_provider} (${data.llm_model})`;
      valVectordb.textContent = data.vector_db_type;
      valChunksCount.textContent = data.total_indexed_chunks;
      valEmbedding.textContent = data.embedding_provider;
      systemStatusText.textContent = "Operational";

      await refreshSources();
    } catch (err) {
      console.error(err);
      systemStatusText.textContent = "Connection Error";
    }
  }

  async function refreshSources() {
    try {
      const res = await fetch("/api/v1/documents/stats");
      if (!res.ok) return;
      const stats = await res.json();

      sourceCount.textContent = stats.unique_sources.length;
      sourceList.innerHTML = "";

      if (stats.unique_sources.length === 0) {
        sourceList.innerHTML = '<li class="empty-hint">No documents indexed yet</li>';
      } else {
        stats.unique_sources.forEach(src => {
          const li = document.createElement("li");
          li.textContent = src;
          li.title = src;
          sourceList.appendChild(li);
        });
      }
    } catch (err) {
      console.error("Failed to load sources:", err);
    }
  }

  // 2. Sample Chips Click
  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) {
      const q = e.target.getAttribute("data-q");
      if (q) {
        queryInput.value = q;
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  // 3. Seed Sample Documents
  btnSeedSamples.addEventListener("click", async () => {
    btnSeedSamples.disabled = true;
    btnSeedSamples.textContent = "⏳ Seeding docs...";
    try {
      const res = await fetch("/api/v1/documents/index-samples", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to seed");
      appendSystemNotification(`📚 ${data.message}`);
      await refreshSystemStatus();
    } catch (err) {
      alert("Error seeding sample docs: " + err.message);
    } finally {
      btnSeedSamples.disabled = false;
      btnSeedSamples.innerHTML = "<span>📚</span> Seed Sample Docs";
    }
  });

  // 4. Clear Index
  btnClearIndex.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to reset and clear all indexed documents?")) return;
    try {
      const res = await fetch("/api/v1/documents", { method: "DELETE" });
      if (!res.ok) throw new Error("Clear failed");
      chatHistory.length = 0;
      appendSystemNotification("🗑️ Knowledge base cleared.");
      await refreshSystemStatus();
    } catch (err) {
      alert("Error clearing knowledge base: " + err.message);
    }
  });

  // 5. Ingest Modal handlers
  btnOpenIngest.addEventListener("click", () => ingestModal.classList.remove("hidden"));
  modalClose.addEventListener("click", () => ingestModal.classList.add("hidden"));
  btnCancelIngest.addEventListener("click", () => ingestModal.classList.add("hidden"));

  ingestForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = document.getElementById("doc-title").value.trim();
    const content = document.getElementById("doc-content").value.trim();
    if (!title || !content) return;

    try {
      const res = await fetch("/api/v1/documents/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Ingest failed");

      ingestModal.classList.add("hidden");
      ingestForm.reset();
      appendSystemNotification(`✅ Ingested: "${title}" (${data.total_chunks_created} chunks created)`);
      await refreshSystemStatus();
    } catch (err) {
      alert("Ingestion error: " + err.message);
    }
  });

  // 6. Chat Submission
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    queryInput.value = "";
    appendUserMessage(query);

    const isStreaming = streamToggle.checked;
    if (isStreaming) {
      await handleStreamingQuery(query);
    } else {
      await handleStandardQuery(query);
    }
  });

  function appendUserMessage(text) {
    chatHistory.push({ role: "user", content: text });
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user";
    msgDiv.innerHTML = `
      <div class="avatar">👤</div>
      <div class="content">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendSystemNotification(text) {
    const note = document.createElement("div");
    note.className = "message-meta";
    note.style.textAlign = "center";
    note.style.justifyContent = "center";
    note.textContent = text;
    chatMessages.appendChild(note);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // 7. Standard JSON Query
  async function handleStandardQuery(query) {
    const assistantDiv = createAssistantMessageElement();
    const contentDiv = assistantDiv.querySelector(".content-text");
    contentDiv.textContent = "Retrieving context & generating answer...";

    try {
      const res = await fetch("/api/v1/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          chat_history: chatHistory.slice(-6)
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Query failed");

      contentDiv.innerHTML = renderMarkdown(data.answer);
      chatHistory.push({ role: "assistant", content: data.answer });

      const meta = assistantDiv.querySelector(".message-meta");
      meta.textContent = `⚡ Latency: ${data.execution_time_ms} ms | Model: ${data.model_used}`;
      await refreshSystemStatus();
    } catch (err) {
      contentDiv.textContent = "Error: " + err.message;
    } finally {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  // 8. Streaming Query via SSE
  async function handleStreamingQuery(query) {
    const assistantDiv = createAssistantMessageElement();
    const contentDiv = assistantDiv.querySelector(".content-text");
    contentDiv.textContent = "";

    let accumulatedText = "";

    try {
      const response = await fetch("/api/v1/rag/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          chat_history: chatHistory.slice(-6)
        })
      });

      if (!response.ok) throw new Error("Streaming connection failed");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep last unfinished piece

        for (const block of lines) {
          if (!block.trim()) continue;
          let eventType = "message";
          let dataStr = "";

          const linesInBlock = block.split("\n");
          for (const line of linesInBlock) {
            if (line.startsWith("event:")) {
              eventType = line.replace("event:", "").trim();
            } else if (line.startsWith("data:")) {
              dataStr = line.replace("data:", "").trim();
            }
          }

          if (eventType === "token") {
            try {
              const token = JSON.parse(dataStr);
              accumulatedText += token;
              contentDiv.innerHTML = renderMarkdown(accumulatedText);
              chatMessages.scrollTop = chatMessages.scrollHeight;
            } catch (e) {
              accumulatedText += dataStr;
              contentDiv.innerHTML = renderMarkdown(accumulatedText);
            }
          } else if (eventType === "error") {
            try {
              const errObj = JSON.parse(dataStr);
              accumulatedText = `❌ Error: ${errObj.error || dataStr}`;
              contentDiv.innerHTML = `<span style="color:#ef4444">${escapeHtml(accumulatedText)}</span>`;
            } catch (e) {
              accumulatedText = `❌ Error: ${dataStr}`;
              contentDiv.innerHTML = `<span style="color:#ef4444">${escapeHtml(accumulatedText)}</span>`;
            }
          } else if (eventType === "done") {
            try {
              const doneMeta = JSON.parse(dataStr);
              const meta = assistantDiv.querySelector(".message-meta");
              meta.textContent = `⚡ Latency: ${doneMeta.execution_time_ms} ms | Model: ${doneMeta.model_used}`;
            } catch (e) {}
          }
        }
      }

      if (!accumulatedText.trim()) {
        contentDiv.innerHTML = `<span style="color:#ef4444">⚠️ No response received from model. Please verify your model name and API key in .env.</span>`;
      }

      chatHistory.push({ role: "assistant", content: accumulatedText });
      await refreshSystemStatus();
    } catch (err) {
      contentDiv.textContent = "Streaming error: " + err.message;
    } finally {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  function createAssistantMessageElement() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    msgDiv.innerHTML = `
      <div class="avatar">🤖</div>
      <div class="content">
        <div class="content-text markdown-body"></div>
        <div class="message-meta"></div>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
  }

  // Markdown Parser (supports marked.js or built-in fallback)
  function renderMarkdown(text) {
    if (!text) return "";
    if (window.marked && typeof window.marked.parse === "function") {
      try {
        return window.marked.parse(text);
      } catch (e) {
        console.warn("marked.js parse fallback", e);
      }
    }
    return parseBasicMarkdown(text);
  }

  function parseBasicMarkdown(text) {
    let html = escapeHtml(text);
    // Code blocks
    html = html.replace(/```([a-z0-9]*)\n([\s\S]*?)```/gi, '<pre><code>$2</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Headings
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    // Bullet list items
    html = html.replace(/^\s*[-*]\s+(.*)$/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gis, '<ul>$1</ul>');
    // Line breaks
    html = html.replace(/\n\n+/g, '</p><p>');
    return `<p>${html}</p>`.replace(/<p><\/p>/g, '');
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
  }

  // Initial load
  refreshSystemStatus();
});
