const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const conversationList = document.getElementById("conversation-list");
const statusText = document.getElementById("status-text");
const statusDot = document.getElementById("status-dot");
const newChatButton = document.getElementById("new-chat");
const newChatTop = document.getElementById("new-chat-top");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const openSidebarButton = document.getElementById("open-sidebar");
const closeSidebarButton = document.getElementById("close-sidebar");
const toast = document.getElementById("toast");

let conversationId = null;
let busy = false;
let toastTimer = null;
let typingNode = null;
const sessions = [];

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2800);
}

function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("visible");
}

function openSidebar() {
  sidebar.classList.add("open");
  sidebarOverlay.classList.add("visible");
}

openSidebarButton.addEventListener("click", openSidebar);
closeSidebarButton.addEventListener("click", closeSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);

function scrollToBottom() {
  const scroller = document.getElementById("chat-scroll");
  requestAnimationFrame(() => {
    scroller.scrollTop = scroller.scrollHeight;
  });
}

function hideWelcome() {
  if (welcome && welcome.isConnected) welcome.remove();
}

function makeMessage(role, text, options = {}) {
  hideWelcome();

  const row = document.createElement("article");
  row.className = `message-row ${role === "user" ? "user-row" : "assistant-row"}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "T" : "F";
  row.appendChild(avatar);

  const body = document.createElement("div");
  body.className = "message-body";

  const name = document.createElement("div");
  name.className = "message-name";
  name.textContent = role === "user" ? "Toi" : "Assistant";
  body.appendChild(name);

  const paragraph = document.createElement("p");
  paragraph.className = "message-text";
  paragraph.textContent = text;
  body.appendChild(paragraph);

  if (options.fon || options.audioUrl || options.warnings?.length) {
    const panel = document.createElement("section");
    panel.className = "fon-panel";

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "fon-toggle";
    toggle.setAttribute("aria-expanded", "false");

    const label = document.createElement("span");
    label.textContent = "Traduction en Fon";
    const arrow = document.createElement("span");
    arrow.textContent = "＋";
    toggle.append(label, arrow);

    const content = document.createElement("div");
    content.className = "fon-content";
    content.hidden = true;

    toggle.addEventListener("click", () => {
      content.hidden = !content.hidden;
      toggle.setAttribute("aria-expanded", String(!content.hidden));
      arrow.textContent = content.hidden ? "＋" : "−";
    });

    if (options.fon) {
      const caption = document.createElement("div");
      caption.className = "fon-caption";
      caption.textContent = "FONGBE";
      const fonText = document.createElement("p");
      fonText.className = "message-text";
      fonText.textContent = options.fon;
      content.append(caption, fonText);
    } else {
      const noTranslation = document.createElement("p");
      noTranslation.className = "message-text";
      noTranslation.textContent = "La traduction n'est pas disponible pour cette réponse.";
      content.appendChild(noTranslation);
    }

    if (options.audioUrl) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = options.audioUrl;
      audio.setAttribute("aria-label", "Écouter la réponse en Fon");
      audio.addEventListener("error", () => {
        addWarning(content, "Le fichier audio n'a pas pu être lu.");
      }, { once: true });
      content.appendChild(audio);
    }

    for (const warning of (options.warnings || [])) {
      addWarning(content, warning);
    }

    panel.append(toggle, content);
    body.appendChild(panel);
  }

  row.appendChild(body);
  messages.appendChild(row);
  scrollToBottom();
  return row;
}

function addWarning(parent, message) {
  const warning = document.createElement("p");
  warning.className = "warning";
  warning.textContent = message;
  parent.appendChild(warning);
}

function showTyping() {
  hideWelcome();
  typingNode = document.createElement("div");
  typingNode.className = "message-row assistant-row";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "F";

  const body = document.createElement("div");
  body.className = "message-body";

  const name = document.createElement("div");
  name.className = "message-name";
  name.textContent = "Assistant";

  const dots = document.createElement("div");
  dots.className = "typing";
  dots.setAttribute("aria-label", "Réponse en cours");
  for (let i = 0; i < 3; i++) dots.appendChild(document.createElement("span"));

  body.append(name, dots);
  typingNode.append(avatar, body);
  messages.appendChild(typingNode);
  scrollToBottom();
}

function hideTyping() {
  if (typingNode) typingNode.remove();
  typingNode = null;
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  input.disabled = value;
  sendButton.setAttribute("aria-label", value ? "Réponse en cours" : "Envoyer le message");
}

function renderHistory() {
  conversationList.replaceChildren();

  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.className = "empty-history";
    empty.textContent = "Tes conversations apparaîtront ici.";
    conversationList.appendChild(empty);
    return;
  }

  for (const session of [...sessions].reverse()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item${session.id === conversationId ? " active" : ""}`;

    const symbol = document.createElement("span");
    symbol.className = "history-symbol";
    symbol.textContent = "▤";

    const title = document.createElement("span");
    title.textContent = session.title || "Nouvelle conversation";
    button.append(symbol, title);

    button.addEventListener("click", () => {
      if (busy) {
        showToast("Attends la fin de la réponse avant de changer de conversation.");
        return;
      }
      switchConversation(session.id);
      closeSidebar();
    });

    conversationList.appendChild(button);
  }
}

function currentSession() {
  return sessions.find((item) => item.id === conversationId);
}

function switchConversation(id) {
  conversationId = id;
  const session = currentSession();
  messages.replaceChildren();

  if (!session) {
    conversationId = null;
    renderWelcome();
    renderHistory();
    return;
  }

  for (const message of session.messages) {
    makeMessage(message.role, message.text, message.options || {});
  }
  renderHistory();
  scrollToBottom();
}

function renderWelcome() {
  messages.innerHTML = `
    <section id="welcome" class="welcome">
      <div class="welcome-logo">F</div>
      <h1>Que souhaites-tu apprendre ?</h1>
      <p>Pose une question en français. Je peux te répondre, traduire en Fon et essayer de prononcer la traduction.</p>
      <div class="suggestions">
        <button class="suggestion" type="button" data-prompt="Bonjour, présente-toi et explique ce que tu peux faire."><span class="suggestion-icon">✦</span><span><strong>Faire connaissance</strong><small>Découvrir l'assistant</small></span></button>
        <button class="suggestion" type="button" data-prompt="Comment dit-on « bonjour » en Fon ?"><span class="suggestion-icon">文</span><span><strong>Découvrir le Fon</strong><small>Explorer une traduction</small></span></button>
        <button class="suggestion" type="button" data-prompt="Explique-moi simplement ce qu'est l'intelligence artificielle."><span class="suggestion-icon">◈</span><span><strong>Comprendre un sujet</strong><small>Obtenir une explication</small></span></button>
        <button class="suggestion" type="button" data-prompt="Combien font 8 + 2 ?"><span class="suggestion-icon">∑</span><span><strong>Faire un calcul</strong><small>Essayer une question simple</small></span></button>
      </div>
    </section>`;
}

function startNewConversation() {
  if (busy) {
    showToast("Attends la fin de la réponse avant de commencer une autre conversation.");
    return;
  }
  conversationId = null;
  messages.replaceChildren();
  renderWelcome();
  renderHistory();
  closeSidebar();
  input.focus();
}

newChatButton.addEventListener("click", startNewConversation);
newChatTop.addEventListener("click", startNewConversation);

document.addEventListener("click", (event) => {
  const suggestion = event.target.closest("[data-prompt]");
  if (!suggestion || busy) return;
  input.value = suggestion.dataset.prompt || "";
  input.focus();
  resizeInput();
});

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}
input.addEventListener("input", resizeInput);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;

  const question = input.value.trim();
  if (!question) return;

  if (!conversationId) {
    conversationId = crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    sessions.push({
      id: conversationId,
      title: question.length > 35 ? `${question.slice(0, 35)}…` : question,
      messages: []
    });
  }

  const session = currentSession();
  if (!session) {
    showToast("Impossible d'ouvrir cette conversation. Crée-en une nouvelle.");
    return;
  }

  session.messages.push({ role: "user", text: question });
  makeMessage("user", question);
  input.value = "";
  resizeInput();
  setBusy(true);
  showTyping();
  renderHistory();

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 360000);

    let response;
    try {
      response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          conversation_id: conversationId
        }),
        signal: controller.signal
      });
    } finally {
      clearTimeout(timeout);
    }

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("Le serveur a renvoyé une réponse illisible.");
    }

    if (!response.ok) {
      throw new Error(data.error || `Erreur du serveur (${response.status}).`);
    }

    const options = {
      fon: data.answer_fon || null,
      audioUrl: data.audio_url || null,
      warnings: data.warnings || []
    };
    const answer = data.answer_fr || "Aucune réponse française n'a été reçue.";

    session.messages.push({ role: "assistant", text: answer, options });
    hideTyping();
    makeMessage("assistant", answer, options);

  } catch (error) {
    hideTyping();

    let message = error.message || "Une erreur inattendue est survenue.";
    if (error.name === "AbortError") {
      message = "Le délai d'attente est dépassé. Le traitement peut être très long sur cet ordinateur. Tu peux réessayer.";
    }

    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    const card = document.createElement("div");
    card.className = "error-card";
    card.textContent = `${message} Ta question reste dans la conversation : tu peux réessayer.`;
    row.appendChild(card);
    messages.appendChild(row);
    scrollToBottom();
    showToast("La réponse n'a pas pu être obtenue.");
  } finally {
    setBusy(false);
    input.focus();
    renderHistory();
    checkHealth();
  }
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Serveur indisponible");
    const health = await response.json();
    const ready = health.ollama === "ok";

    statusDot.className = `status-dot ${ready ? "good" : "bad"}`;
    statusText.textContent = ready
      ? "Assistant prêt"
      : `Ollama : ${health.ollama}. Vérifie qu'il est lancé.`;

    const indicator = document.querySelector(".status-indicator");
    indicator.className = `status-indicator ${ready ? "good" : "bad"}`;
    indicator.title = ready ? "Ollama disponible" : "Ollama indisponible";
  } catch {
    statusDot.className = "status-dot bad";
    statusText.textContent = "Serveur indisponible";
    const indicator = document.querySelector(".status-indicator");
    indicator.className = "status-indicator bad";
  }
}

renderHistory();
checkHealth();
setInterval(checkHealth, 30000);
