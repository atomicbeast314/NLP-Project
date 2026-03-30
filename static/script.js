const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const clearBtn = document.getElementById("clear-btn");

// ── Send message ────────────────────────────────────────────────────────────

chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;
    sendMessage(text);
});

async function sendMessage(text) {
    // Remove suggestion chips on first user message
    const suggestionsEl = document.getElementById("suggestions");
    if (suggestionsEl) suggestionsEl.remove();

    // Add user bubble
    appendMessage("user", text);
    userInput.value = "";
    userInput.focus();

    // Show typing indicator
    const typingEl = showTyping();

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });
        const data = await res.json();

        // Remove typing indicator
        typingEl.remove();

        // Add bot response
        appendBotMessage(data.response, data.intent, data.confidence);
    } catch (err) {
        typingEl.remove();
        appendBotMessage(
            "Sorry, something went wrong. Please make sure the server is running.",
            "error",
            0
        );
    }

    scrollToBottom();
}

// ── Append messages ─────────────────────────────────────────────────────────

function appendMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `msg ${role}-msg`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${role}-avatar`;
    avatar.textContent = role === "bot" ? "NLP" : "\u{1f464}";

    const bubble = document.createElement("div");
    bubble.className = `bubble ${role}-bubble`;
    bubble.textContent = text;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    messagesEl.appendChild(msgDiv);
    scrollToBottom();
}

function appendBotMessage(text, intent, confidence) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "msg bot-msg";

    const avatar = document.createElement("div");
    avatar.className = "avatar bot-avatar";
    avatar.textContent = "NLP";

    const bubble = document.createElement("div");
    bubble.className = "bubble bot-bubble";

    const p = document.createElement("p");
    p.textContent = text;
    bubble.appendChild(p);

    // Intent tag
    const tag = document.createElement("div");
    tag.className = `intent-tag${intent === "unknown" ? " unknown" : ""}`;

    const intentLabel = intent.replace(/_/g, " ");
    const pct = Math.round(confidence * 100);

    tag.innerHTML = `
        <span>${intentLabel}</span>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${pct}%"></div>
        </div>
        <span>${pct}%</span>
    `;
    bubble.appendChild(tag);

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    messagesEl.appendChild(msgDiv);
    scrollToBottom();
}

// ── Typing indicator ────────────────────────────────────────────────────────

function showTyping() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "msg bot-msg";

    const avatar = document.createElement("div");
    avatar.className = "avatar bot-avatar";
    avatar.textContent = "NLP";

    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    typing.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(typing);
    messagesEl.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

// ── Scroll ──────────────────────────────────────────────────────────────────

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Suggestion chips & sidebar buttons ──────────────────────────────────────

document.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) {
        sendMessage(chip.dataset.query);
        return;
    }

    const topicBtn = e.target.closest(".topic-btn");
    if (topicBtn) {
        sendMessage(topicBtn.dataset.query);
    }
});

// ── Clear chat ──────────────────────────────────────────────────────────────

clearBtn.addEventListener("click", () => {
    messagesEl.innerHTML = `
        <div class="msg bot-msg">
            <div class="avatar bot-avatar">NLP</div>
            <div class="bubble bot-bubble">
                <p>Chat cleared! Ask me anything about <strong>NLP concepts</strong> or <strong>AI applications</strong>.</p>
            </div>
        </div>
        <div class="suggestions" id="suggestions">
            <button class="chip" data-query="What is NLP?">What is NLP?</button>
            <button class="chip" data-query="Explain tokenization">Explain tokenization</button>
            <button class="chip" data-query="How does sentiment analysis work?">Sentiment analysis</button>
            <button class="chip" data-query="What is BERT?">What is BERT?</button>
            <button class="chip" data-query="AI applications in healthcare">AI in healthcare</button>
            <button class="chip" data-query="What are word embeddings?">Word embeddings</button>
        </div>
    `;
});
