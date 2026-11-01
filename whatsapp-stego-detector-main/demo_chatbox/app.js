/**
 * WhatsApp Steganography Detector - Authentic Web Interface
 * Integrates local ML steganalysis with authentic WhatsApp Web UX
 */

const BACKEND_URL = "http://127.0.0.1:5000/predict";
const HEALTH_URL = "http://127.0.0.1:5000/health";

// Academic Model Metrics from metrics.json for viva presentation (8 Parameters)
const MODEL_METRICS = {
  epochs: 25,
  batch_size: 32,
  seed: 42,
  algorithm: "Random Forest Ensemble (300 Trees) & Deep MLP Neural Network (128-64-32)",
  algorithm_details: {
    primary_model: "Random Forest Classifier (300 Estimators, Gini Impurity, Class-Balanced)",
    neural_network: "Multi-Layer Perceptron (MLP) Deep Steganalyzer",
    architecture: "Input: 24 Features -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(2, Softmax)",
    optimizer: "Adam (Adaptive Moment Estimation, lr=0.001, beta1=0.9, beta2=0.999)",
    loss_function: "Categorical Cross-Entropy / Log Loss",
    features_engineered: "24 Classical Steganalysis Statistical Features (Chi-Square PoV, RS Analysis, Shannon Entropy, SPA)"
  },
  accuracy: 95.83,
  roc_auc: 0.9931,
  precision: 95.83,
  recall: 95.83,
  specificity: 95.83,
  f1: 95.83,
  cv_f1_mean: 97.35,
  cv_f1_std: 0.0242,
  n_train: 192,
  n_test: 48,
  confusion_matrix: {
    matrix: [[23, 1], [1, 23]],
    true_negative: 23,
    false_positive: 1,
    false_negative: 1,
    true_positive: 23,
    fp_rate: 0.0417,
    fn_rate: 0.0417,
    accuracy: 0.9583,
    precision: 0.9583,
    recall: 0.9583,
    specificity: 0.9583,
    f1: 0.9583
  },
  roc_curve: {
    auc: 0.9931,
    threshold_recommended: 0.50,
    operating_tpr: 0.9583,
    operating_fpr: 0.0417,
    points: [
      { fpr: 0.0, tpr: 0.0, threshold: 1.0 },
      { fpr: 0.0, tpr: 0.0417, threshold: 0.9967 },
      { fpr: 0.0, tpr: 0.9583, threshold: 0.665 },
      { fpr: 0.1667, tpr: 0.9583, threshold: 0.2714 },
      { fpr: 0.1667, tpr: 1.0, threshold: 0.2694 },
      { fpr: 1.0, tpr: 1.0, threshold: 0.0025 }
    ]
  },
  epoch_history: [
    { epoch: 1, loss: 0.6529, val_loss: 0.5270, train_accuracy: 79.38, validation_accuracy: 84.38, testing_accuracy: 77.08 },
    { epoch: 2, loss: 0.5130, val_loss: 0.4062, train_accuracy: 92.50, validation_accuracy: 90.62, testing_accuracy: 93.75 },
    { epoch: 3, loss: 0.3851, val_loss: 0.2922, train_accuracy: 96.25, validation_accuracy: 93.75, testing_accuracy: 95.83 },
    { epoch: 4, loss: 0.2710, val_loss: 0.2045, train_accuracy: 96.25, validation_accuracy: 93.75, testing_accuracy: 95.83 },
    { epoch: 5, loss: 0.1801, val_loss: 0.1437, train_accuracy: 96.25, validation_accuracy: 93.75, testing_accuracy: 95.83 },
    { epoch: 6, loss: 0.1196, val_loss: 0.1069, train_accuracy: 96.88, validation_accuracy: 93.75, testing_accuracy: 95.83 },
    { epoch: 7, loss: 0.0862, val_loss: 0.0873, train_accuracy: 96.88, validation_accuracy: 93.75, testing_accuracy: 95.83 },
    { epoch: 8, loss: 0.0717, val_loss: 0.0781, train_accuracy: 96.88, validation_accuracy: 93.75, testing_accuracy: 95.83 },
    { epoch: 9, loss: 0.0673, val_loss: 0.0747, train_accuracy: 96.88, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 10, loss: 0.0660, val_loss: 0.0738, train_accuracy: 96.88, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 11, loss: 0.0673, val_loss: 0.0746, train_accuracy: 97.50, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 12, loss: 0.0694, val_loss: 0.0759, train_accuracy: 97.50, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 13, loss: 0.0722, val_loss: 0.0772, train_accuracy: 98.12, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 14, loss: 0.0754, val_loss: 0.0779, train_accuracy: 98.12, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 15, loss: 0.0782, val_loss: 0.0778, train_accuracy: 98.12, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 16, loss: 0.0792, val_loss: 0.0768, train_accuracy: 98.12, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 17, loss: 0.0778, val_loss: 0.0755, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 18, loss: 0.0743, val_loss: 0.0746, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 19, loss: 0.0701, val_loss: 0.0746, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 20, loss: 0.0665, val_loss: 0.0751, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 21, loss: 0.0632, val_loss: 0.0752, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 22, loss: 0.0594, val_loss: 0.0735, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 23, loss: 0.0551, val_loss: 0.0694, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 95.83 },
    { epoch: 24, loss: 0.0511, val_loss: 0.0630, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 93.75 },
    { epoch: 25, loss: 0.0489, val_loss: 0.0549, train_accuracy: 98.75, validation_accuracy: 96.88, testing_accuracy: 93.75 }
  ],
  top_features: [
    { name: "G_chi_square", weight: "14.86%", desc: "Chi-Square POV attack on Green channel (flags pairs of values parity flattening)" },
    { name: "R_rs_rate", weight: "12.99%", desc: "Regular & Singular groups variation under mask on Red channel" },
    { name: "B_lsb_entropy", weight: "12.92%", desc: "Shannon entropy of LSB bitplane in Blue channel (randomness spike)" },
    { name: "B_rs_rate", weight: "11.61%", desc: "RS Steganalysis boundary shift rate on Blue channel" },
    { name: "R_lsb_entropy", weight: "9.86%", desc: "Shannon entropy of LSB bitplane in Red channel" },
    { name: "G_lsb_entropy", weight: "8.61%", desc: "Shannon entropy of LSB bitplane in Green channel" },
    { name: "G_rs_rate", weight: "7.47%", desc: "RS Steganalysis flip variation on Green channel" },
    { name: "G_lsb_ones_ratio", weight: "4.49%", desc: "Ratio of 1s to 0s in LSB plane (LSB substitution tends toward 0.5)" }
  ]
};

// Chat Contacts Data
const CHAT_DATA = {
  alex: {
    id: "alex",
    name: "Alex Rivera (PBL Team)",
    status: "online",
    avatarClass: "alex",
    avatarText: "AR",
    initialMessages: [
      { type: "text", direction: "in", text: "Hey! Did you add the 8 evaluation parameters mam asked for? (Epochs, Batch size, Seed, Confusion Matrix, Accuracy curves, Loss per epoch, Algorithm, and ROC curve?)", time: "10:30 AM" },
      { type: "text", direction: "out", text: "Yes! Integrated all 8 parameters into the pipeline & dashboard: 25 Epochs, Batch Size 32, Seed 42, Random Forest + Deep MLP, 95.83% Accuracy, and 0.9931 ROC-AUC.", time: "10:32 AM" },
      { type: "text", direction: "in", text: "Awesome! Click the shield pill on the top right or 'View Model & Viva Metrics' anytime to show mam the full dashboard with all curves & matrix.", time: "10:33 AM" }
    ]
  },
  lab: {
    id: "lab",
    name: "Cyber Security Lab 🛡️",
    status: "Prof. Patel, Alex, You",
    avatarClass: "lab",
    avatarText: "CS",
    initialMessages: [
      { type: "text", direction: "in", text: "Prof. Patel: Remember for the viva that WhatsApp re-encodes images as JPEG on normal sends.", time: "09:12 AM" },
      { type: "text", direction: "in", text: "Alex: But WhatsApp Web Document Mode and View-Once pass images without lossy spatial destruction.", time: "09:14 AM" },
      { type: "text", direction: "out", text: "Exactly! And our demo app inspects byte-exact images to verify pure LSB steganalysis.", time: "09:15 AM" }
    ]
  },
  prof: {
    id: "prof",
    name: "Prof. Sharma (Evaluation)",
    status: "last seen today at 10:15 AM",
    avatarClass: "prof",
    avatarText: "PS",
    initialMessages: [
      { type: "text", direction: "in", text: "Hello team. Please demonstrate the detector's classification on both clean and stego samples.", time: "Yesterday" },
      { type: "text", direction: "out", text: "Good morning Professor! You can send any image via 📎 or click the quick demo buttons to see real-time inference and LSB bitplanes.", time: "Yesterday" }
    ]
  },
  saved: {
    id: "saved",
    name: "You (Saved Payloads)",
    status: "Message yourself",
    avatarClass: "clean",
    avatarText: "ME",
    initialMessages: [
      { type: "text", direction: "out", text: "Notes for viva: Chi-Square tests pairs of values (PoVs) parity. RS analysis tests mask variation. LSB entropy tests bit randomness.", time: "Sunday" }
    ]
  }
};

let currentChatId = "alex";
let currentImageData = null; // For forensic modal inspector
let currentSender = "out"; // "out" = You (Outgoing), "in" = Friend (Incoming)

// Helper to compute initials from contact name
function getInitials(name) {
  const clean = name.replace(/[^\w\s]/gi, '').trim();
  const parts = clean.split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase();
  } else if (parts.length === 1 && parts[0].length >= 2) {
    return parts[0].slice(0, 2).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase() || "WA";
}

// Load custom contact names from localStorage
Object.keys(CHAT_DATA).forEach((k) => {
  const saved = localStorage.getItem("wa_contact_name_" + k);
  if (saved) {
    CHAT_DATA[k].name = saved;
    CHAT_DATA[k].avatarText = getInitials(saved);
  }
});

// Update sidebar items on initial load with saved names
function refreshSidebarContacts() {
  Object.keys(CHAT_DATA).forEach((k) => {
    const item = document.querySelector(`.chat-item[data-chat="${k}"]`);
    if (item) {
      const nameEl = item.querySelector(".chat-name");
      if (nameEl) nameEl.textContent = CHAT_DATA[k].name;
      const avatarEl = item.querySelector(".chat-avatar");
      if (avatarEl) {
        const ind = avatarEl.querySelector(".online-indicator");
        avatarEl.textContent = CHAT_DATA[k].avatarText;
        if (ind) avatarEl.appendChild(ind);
      }
    }
  });
}
setTimeout(refreshSidebarContacts, 50);

// DOM Elements
const messagesContainer = document.getElementById("chatMessages");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendActionBtn");
const attachBtn = document.getElementById("attachBtn");
const attachmentMenu = document.getElementById("attachmentMenu");
const fileInput = document.getElementById("fileInput");
const statusPill = document.getElementById("stegoShieldPill");
const shieldText = document.getElementById("shieldStatusText");
const dropOverlay = document.getElementById("dropzoneOverlay");
const emojiBtn = document.getElementById("emojiBtn");
const emojiPopover = document.getElementById("emojiPopover");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const viewModeBtn = document.getElementById("viewModeBtn");
const menuToggleBtn = document.getElementById("menuToggleBtn");
const headerDropdown = document.getElementById("headerDropdown");
const editContactBtn = document.getElementById("editContactBtn");
const senderToggleBtn = document.getElementById("senderToggleBtn");
const senderToggleIcon = document.getElementById("senderToggleIcon");
const senderToggleText = document.getElementById("senderToggleText");

// Forensic Modal Elements
const forensicModal = document.getElementById("forensicModal");
const closeModalBtn = document.getElementById("closeForensicModal");
const modalImgTitle = document.getElementById("modalImgTitle");
const verdictBanner = document.getElementById("verdictBanner");
const verdictTitle = document.getElementById("verdictTitle");
const verdictSub = document.getElementById("verdictSub");
const verdictConfidence = document.getElementById("verdictConfidence");
const probBarSafe = document.getElementById("probBarSafe");
const probBarSuspicious = document.getElementById("probBarSuspicious");
const safePercentText = document.getElementById("safePercentText");
const suspPercentText = document.getElementById("suspPercentText");
const latencyText = document.getElementById("latencyText");
const signalsTableBody = document.getElementById("signalsTableBody");
const bitplaneCanvas = document.getElementById("bitplaneCanvas");
const bitplaneTabs = document.querySelectorAll(".bitplane-tab-btn");

// Metrics Modal Elements
const metricsModal = document.getElementById("metricsModal");
const closeMetricsBtn = document.getElementById("closeMetricsModal");
const openMetricsAction = document.getElementById("openMetricsAction");

// Format Current Time (HH:MM AM/PM)
function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Backend Health Check
function checkBackendHealth() {
  fetch(HEALTH_URL)
    .then((res) => res.json())
    .then((data) => {
      statusPill.className = "stego-shield-pill online";
      shieldText.textContent = data.model_loaded
        ? "Stego Shield Active • Model Ready"
        : "Backend Online (Model Not Loaded)";
    })
    .catch(() => {
      statusPill.className = "stego-shield-pill offline";
      shieldText.textContent = "Backend Offline (Run python app.py)";
    });
}

checkBackendHealth();
setInterval(checkBackendHealth, 4000);

// Double Blue Check SVG Icon
const blueCheckSvg = `
  <svg viewBox="0 0 16 11" fill="currentColor" class="blue-tick">
    <path d="M11.07 0.93a.75.75 0 0 0-1.06 0l-5.72 5.72-2.36-2.36a.75.75 0 0 0-1.06 1.06l2.89 2.89a.75.75 0 0 0 1.06 0l6.25-6.25a.75.75 0 0 0 0-1.06zM15.07 0.93a.75.75 0 0 0-1.06 0l-5.72 5.72-.8-.8a.75.75 0 1 0-1.06 1.06l1.33 1.33a.75.75 0 0 0 1.06 0l6.25-6.25a.75.75 0 0 0 0-1.06z"/>
  </svg>
`;

// Render a Text Message Bubble (Supports editing on hover or double click)
function appendTextMessage(text, direction = "out", time = timeNow(), msgIndex = null) {
  const row = document.createElement("div");
  row.className = `msg-row ${direction}`;

  const bubble = document.createElement("div");
  bubble.className = "wa-bubble";

  const textSpan = document.createElement("span");
  textSpan.className = "msg-text";
  textSpan.textContent = text;
  textSpan.title = "Click ✏️ or double-click to edit this message";

  // Subtle hover edit button
  const editBtn = document.createElement("button");
  editBtn.className = "msg-edit-btn";
  editBtn.title = "Edit message text";
  editBtn.innerHTML = "✏️";

  const startEdit = (e) => {
    if (e) e.stopPropagation();
    textSpan.contentEditable = "true";
    textSpan.focus();
    const range = document.createRange();
    range.selectNodeContents(textSpan);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  };

  editBtn.addEventListener("click", startEdit);
  textSpan.addEventListener("dblclick", startEdit);

  const saveEdit = () => {
    textSpan.contentEditable = "false";
    const newText = textSpan.textContent.trim();
    if (!newText) {
      textSpan.textContent = text;
    } else {
      if (msgIndex !== null && CHAT_DATA[currentChatId] && CHAT_DATA[currentChatId].initialMessages[msgIndex]) {
        CHAT_DATA[currentChatId].initialMessages[msgIndex].text = newText;
        localStorage.setItem("wa_msgs_" + currentChatId, JSON.stringify(CHAT_DATA[currentChatId].initialMessages));
      }
    }
  };

  textSpan.addEventListener("blur", saveEdit);
  textSpan.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      textSpan.blur();
    }
    if (e.key === "Escape") {
      textSpan.textContent = text;
      textSpan.blur();
    }
  });

  const metaSpan = document.createElement("span");
  metaSpan.className = "msg-meta";
  metaSpan.innerHTML = `<span>${time}</span>${direction === "out" ? blueCheckSvg : ""}`;

  bubble.appendChild(textSpan);
  bubble.appendChild(editBtn);
  bubble.appendChild(metaSpan);
  row.appendChild(bubble);
  messagesContainer.appendChild(row);
  scrollToBottom();
}

// Render an Image Message Bubble and trigger Stego Analysis
function appendImageMessage(fileOrSample, direction = "out") {
  const isSample = !fileOrSample.lastModified;
  const fileName = isSample ? fileOrSample.name : fileOrSample.name;
  const fileSizeKb = isSample
    ? (fileOrSample.size / 1024).toFixed(1)
    : (fileOrSample.size / 1024).toFixed(1);

  const processDataUrl = (dataUrl) => {
    const row = document.createElement("div");
    row.className = `msg-row ${direction}`;

    const bubble = document.createElement("div");
    bubble.className = "wa-bubble image-bubble";

    const imgWrapper = document.createElement("div");
    imgWrapper.className = "bubble-image-wrapper";

    const img = document.createElement("img");
    img.src = dataUrl;
    img.alt = fileName;

    const badge = document.createElement("div");
    badge.className = "stego-badge scanning";
    badge.innerHTML = `<span>⏳</span> Scanning…`;

    imgWrapper.appendChild(img);
    imgWrapper.appendChild(badge);

    const metaBar = document.createElement("div");
    metaBar.className = "image-meta-bar";

    const fileInfo = document.createElement("div");
    fileInfo.className = "image-file-info";
    fileInfo.innerHTML = `
      <span><strong>${fileName}</strong> · ${fileSizeKb} kB</span>
      <span style="display:flex;align-items:center;gap:3px">${timeNow()} ${direction === "out" ? blueCheckSvg : ""}</span>
    `;

    const signalsPreview = document.createElement("div");
    signalsPreview.className = "image-signals-preview";
    signalsPreview.textContent = "Analyzing LSB bitplanes & feature distributions…";

    const inspectBtn = document.createElement("div");
    inspectBtn.className = "image-inspect-action";
    inspectBtn.innerHTML = `<span>🔍</span> Click to Inspect Forensics & LSB Bitplanes`;

    metaBar.appendChild(fileInfo);
    metaBar.appendChild(signalsPreview);
    metaBar.appendChild(inspectBtn);

    bubble.appendChild(imgWrapper);
    bubble.appendChild(metaBar);
    row.appendChild(bubble);
    messagesContainer.appendChild(row);
    scrollToBottom();

    // Call Local Steganography Detector Model
    fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    })
      .then((r) => r.json())
      .then((result) => {
        if (result.error) throw new Error(result.error);

        const isSuspicious = result.is_suspicious;
        const confidencePct = Math.round(result.confidence * 100);

        badge.className = `stego-badge ${isSuspicious ? "suspicious" : "safe"}`;
        badge.innerHTML = isSuspicious
          ? `<span>⚠</span> Suspicious (${confidencePct}%)`
          : `<span>✓</span> Safe (${confidencePct}%)`;

        const topSignalsStr = (result.top_signals || [])
          .slice(0, 3)
          .map((s) => `${s.feature}=${s.value}`)
          .join(" · ");

        signalsPreview.textContent = topSignalsStr
          ? `Top signals: ${topSignalsStr} · Latency: ${result.inference_ms}ms`
          : `Prediction: ${result.label.toUpperCase()} (${confidencePct}%)`;

        // Store analysis result on the element for inspector modal
        const forensicData = {
          name: fileName,
          dataUrl: dataUrl,
          sizeKb: fileSizeKb,
          result: result
        };

        const openInspector = () => openForensicInspector(forensicData);
        imgWrapper.addEventListener("click", openInspector);
        inspectBtn.addEventListener("click", openInspector);
      })
      .catch((err) => {
        badge.className = "stego-badge scanning";
        badge.innerHTML = `<span>⚠</span> Backend Offline`;
        signalsPreview.textContent = `Could not reach ${BACKEND_URL}. Ensure 'python app.py' is running.`;
      });
  };

  if (isSample) {
    processDataUrl(fileOrSample.data);
  } else {
    const reader = new FileReader();
    reader.onload = (e) => processDataUrl(e.target.result);
    reader.readAsDataURL(fileOrSample); // Raw byte preservation, zero JPEG compression
  }
}

// Scroll chat container to bottom
function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Switch Active Chat
function switchChat(chatId) {
  if (!CHAT_DATA[chatId]) return;
  currentChatId = chatId;
  const chat = CHAT_DATA[chatId];

  // Update sidebar active class
  document.querySelectorAll(".chat-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.chat === chatId);
  });

  // Update header
  const headerAvatar = document.getElementById("chatHeaderAvatar");
  const headerName = document.getElementById("chatHeaderName");
  const headerStatus = document.getElementById("chatHeaderStatus");

  headerAvatar.className = `chat-header-avatar ${chat.avatarClass}`;
  headerAvatar.textContent = chat.avatarText;
  headerName.textContent = chat.name;
  headerStatus.textContent = chat.status;
  headerStatus.className = `chat-header-status ${chat.status === "online" ? "online-text" : ""}`;

  // Clear messages and load initial
  messagesContainer.innerHTML = "";

  // Add Date Separator
  const dateSep = document.createElement("div");
  dateSep.className = "chat-date-separator";
  dateSep.textContent = "TODAY";
  messagesContainer.appendChild(dateSep);

  // Add Encryption notice
  const encNotice = document.createElement("div");
  encNotice.className = "encryption-notice";
  encNotice.innerHTML = `
    <svg viewBox="0 0 16 16"><path d="M8 1a4 4 0 0 0-4 4v2H3a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-1V5a4 4 0 0 0-4-4zm-2 4a2 2 0 1 1 4 0v2H6V5zm2 5a1.25 1.25 0 0 1 1.25 1.25c0 .48-.27.89-.67 1.1v1.4a.58.58 0 1 1-1.16 0v-1.4a1.25 1.25 0 0 1-.67-1.1A1.25 1.25 0 0 1 8 10z"/></svg>
    <span>Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.</span>
  `;
  messagesContainer.appendChild(encNotice);

  // Load saved messages if any
  const savedMsgs = localStorage.getItem("wa_msgs_" + chatId);
  if (savedMsgs) {
    try {
      chat.initialMessages = JSON.parse(savedMsgs);
    } catch (e) {}
  }

  // Load chat initial messages
  chat.initialMessages.forEach((msg, idx) => {
    if (msg.type === "text") {
      appendTextMessage(msg.text, msg.direction, msg.time, idx);
    }
  });

  scrollToBottom();
}

// =========================================================================
// Forensic Inspector Modal & Interactive LSB Bitplane Visualizer
// =========================================================================
function openForensicInspector(data) {
  currentImageData = data;
  modalImgTitle.textContent = `${data.name} (${data.sizeKb} kB)`;

  const res = data.result;
  const isSuspicious = res.is_suspicious;
  const confPct = Math.round(res.confidence * 100);

  // Verdict Banner
  verdictBanner.className = `verdict-banner ${isSuspicious ? "suspicious" : "safe"}`;
  verdictTitle.textContent = isSuspicious ? "⚠ SUSPICIOUS (Payload Detected)" : "✓ SAFE (Pristine Image)";
  verdictSub.textContent = isSuspicious
    ? "High probability of Least-Significant-Bit (LSB) steganographic modification."
    : "Pixel distributions conform to natural un-embedded photographic statistics.";
  verdictConfidence.textContent = `${confPct}%`;

  // Probability Meter
  const safePct = Math.round(res.probabilities.safe * 100);
  const suspPct = Math.round(res.probabilities.suspicious * 100);
  probBarSafe.style.width = `${safePct}%`;
  probBarSuspicious.style.width = `${suspPct}%`;
  safePercentText.textContent = `Safe: ${safePct}%`;
  suspPercentText.textContent = `Suspicious: ${suspPct}%`;
  latencyText.textContent = `${res.inference_ms} ms`;

  // Top Signals Table
  signalsTableBody.innerHTML = "";
  (res.top_signals || []).forEach((sig) => {
    const tr = document.createElement("tr");
    let expl = "Statistical metric";
    if (sig.feature.includes("chi")) expl = "Chi-Square POV parity flattening test";
    else if (sig.feature.includes("rs")) expl = "Regular & Singular mask flip shift";
    else if (sig.feature.includes("entropy")) expl = "Shannon entropy of LSB bitplane";
    else if (sig.feature.includes("ratio")) expl = "LSB substitution 0/1 density ratio";

    tr.innerHTML = `
      <td><strong>${sig.feature}</strong></td>
      <td><span class="signal-val-badge">${sig.value}</span></td>
      <td>${expl}</td>
    `;
    signalsTableBody.appendChild(tr);
  });

  // Render original in LSB Bitplane Canvas
  renderBitplane("original");

  forensicModal.classList.add("open");
}

// Client-side LSB Bitplane Extraction on HTML5 Canvas
function renderBitplane(mode) {
  if (!currentImageData) return;

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    bitplaneCanvas.width = img.naturalWidth || 200;
    bitplaneCanvas.height = img.naturalHeight || 200;
    const ctx = bitplaneCanvas.getContext("2d");

    ctx.drawImage(img, 0, 0);
    if (mode === "original") return;

    const imgData = ctx.getImageData(0, 0, bitplaneCanvas.width, bitplaneCanvas.height);
    const d = imgData.data;

    for (let i = 0; i < d.length; i += 4) {
      const r = d[i];
      const g = d[i + 1];
      const b = d[i + 2];

      let bitVal = 0;
      if (mode === "red") {
        bitVal = (r & 1) ? 255 : 0;
        d[i] = bitVal;
        d[i + 1] = bitVal;
        d[i + 2] = bitVal;
      } else if (mode === "green") {
        bitVal = (g & 1) ? 255 : 0;
        d[i] = bitVal;
        d[i + 1] = bitVal;
        d[i + 2] = bitVal;
      } else if (mode === "blue") {
        bitVal = (b & 1) ? 255 : 0;
        d[i] = bitVal;
        d[i + 1] = bitVal;
        d[i + 2] = bitVal;
      } else if (mode === "all") {
        // Combined parity: XOR of LSBs
        bitVal = ((r & 1) ^ (g & 1) ^ (b & 1)) ? 255 : 0;
        d[i] = bitVal;
        d[i + 1] = bitVal;
        d[i + 2] = bitVal;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  };
  img.src = currentImageData.dataUrl;
}

// Bitplane Tabs Switcher
bitplaneTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    bitplaneTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    renderBitplane(tab.dataset.plane);
  });
});

closeModalBtn.addEventListener("click", () => forensicModal.classList.remove("open"));
forensicModal.addEventListener("click", (e) => {
  if (e.target === forensicModal) forensicModal.classList.remove("open");
});

// =========================================================================
// Viva / Academic Metrics Modal & Interactive Visualizations
// =========================================================================
function renderAccuracyChart(history) {
  const container = document.getElementById("accuracyChartSvg");
  if (!container || !history || !history.length) return;

  const w = 420;
  const h = 200;
  const padL = 40;
  const padR = 20;
  const padT = 20;
  const padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const minEpoch = 1;
  const maxEpoch = history.length;
  const minAcc = 50;
  const maxAcc = 100;

  const getX = (ep) => padL + ((ep - minEpoch) / (maxEpoch - minEpoch)) * plotW;
  const getY = (acc) => padT + plotH - ((acc - minAcc) / (maxAcc - minAcc)) * plotH;

  // Grid lines
  let gridSvg = "";
  for (let acc = 60; acc <= 100; acc += 10) {
    const y = getY(acc);
    gridSvg += `
      <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="2,3"/>
      <text x="${padL - 8}" y="${y + 4}" fill="var(--wa-text-secondary)" font-size="10" text-anchor="end" font-family="monospace">${acc}%</text>
    `;
  }

  // X axis labels
  [1, 5, 10, 15, 20, 25].forEach((ep) => {
    if (ep <= maxEpoch) {
      const x = getX(ep);
      gridSvg += `
        <line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.05)"/>
        <text x="${x}" y="${h - 10}" fill="var(--wa-text-secondary)" font-size="10" text-anchor="middle" font-family="monospace">E${ep}</text>
      `;
    }
  });

  const trainPts = history.map((h) => `${getX(h.epoch).toFixed(1)},${getY(h.train_accuracy).toFixed(1)}`).join(" ");
  const valPts = history.map((h) => `${getX(h.epoch).toFixed(1)},${getY(h.validation_accuracy).toFixed(1)}`).join(" ");
  const testPts = history.map((h) => `${getX(h.epoch).toFixed(1)},${getY(h.testing_accuracy).toFixed(1)}`).join(" ");

  let dotsSvg = "";
  history.forEach((h) => {
    const x = getX(h.epoch);
    dotsSvg += `
      <circle cx="${x}" cy="${getY(h.train_accuracy)}" r="3" fill="#10b981">
        <title>Epoch ${h.epoch}: Train Acc ${h.train_accuracy}%</title>
      </circle>
      <circle cx="${x}" cy="${getY(h.validation_accuracy)}" r="3" fill="#06b6d4">
        <title>Epoch ${h.epoch}: Val Acc ${h.validation_accuracy}%</title>
      </circle>
      <circle cx="${x}" cy="${getY(h.testing_accuracy)}" r="3" fill="#8b5cf6">
        <title>Epoch ${h.epoch}: Test Acc ${h.testing_accuracy}%</title>
      </circle>
    `;
  });

  container.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}">
      ${gridSvg}
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="var(--wa-border)" stroke-width="1"/>
      <line x1="${padL}" y1="${padT + plotH}" x2="${w - padR}" y2="${padT + plotH}" stroke="var(--wa-border)" stroke-width="1"/>
      <polyline fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" points="${trainPts}"/>
      <polyline fill="none" stroke="#06b6d4" stroke-width="2" stroke-linejoin="round" stroke-dasharray="3,2" points="${valPts}"/>
      <polyline fill="none" stroke="#8b5cf6" stroke-width="2" stroke-linejoin="round" points="${testPts}"/>
      ${dotsSvg}
    </svg>
  `;
}

function renderLossChart(history) {
  const container = document.getElementById("lossChartSvg");
  if (!container || !history || !history.length) return;

  const w = 420;
  const h = 200;
  const padL = 40;
  const padR = 20;
  const padT = 20;
  const padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const minEpoch = 1;
  const maxEpoch = history.length;
  const minLoss = 0.0;
  const maxLoss = 0.70;

  const getX = (ep) => padL + ((ep - minEpoch) / (maxEpoch - minEpoch)) * plotW;
  const getY = (loss) => padT + plotH - ((loss - minLoss) / (maxLoss - minLoss)) * plotH;

  let gridSvg = "";
  for (let l = 0.1; l <= 0.7; l += 0.15) {
    const y = getY(l);
    gridSvg += `
      <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="2,3"/>
      <text x="${padL - 8}" y="${y + 4}" fill="var(--wa-text-secondary)" font-size="10" text-anchor="end" font-family="monospace">${l.toFixed(2)}</text>
    `;
  }

  [1, 5, 10, 15, 20, 25].forEach((ep) => {
    if (ep <= maxEpoch) {
      const x = getX(ep);
      gridSvg += `
        <line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.05)"/>
        <text x="${x}" y="${h - 10}" fill="var(--wa-text-secondary)" font-size="10" text-anchor="middle" font-family="monospace">E${ep}</text>
      `;
    }
  });

  const trainPts = history.map((h) => `${getX(h.epoch).toFixed(1)},${getY(h.loss).toFixed(1)}`).join(" ");
  const valPts = history.map((h) => `${getX(h.epoch).toFixed(1)},${getY(h.val_loss).toFixed(1)}`).join(" ");

  let dotsSvg = "";
  history.forEach((h) => {
    const x = getX(h.epoch);
    dotsSvg += `
      <circle cx="${x}" cy="${getY(h.loss)}" r="3" fill="#ef4444">
        <title>Epoch ${h.epoch}: Train Loss ${h.loss}</title>
      </circle>
      <circle cx="${x}" cy="${getY(h.val_loss)}" r="3" fill="#f97316">
        <title>Epoch ${h.epoch}: Val Loss ${h.val_loss}</title>
      </circle>
    `;
  });

  container.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}">
      ${gridSvg}
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="var(--wa-border)" stroke-width="1"/>
      <line x1="${padL}" y1="${padT + plotH}" x2="${w - padR}" y2="${padT + plotH}" stroke="var(--wa-border)" stroke-width="1"/>
      <polyline fill="none" stroke="#ef4444" stroke-width="2.5" stroke-linejoin="round" points="${trainPts}"/>
      <polyline fill="none" stroke="#f97316" stroke-width="2" stroke-linejoin="round" stroke-dasharray="3,2" points="${valPts}"/>
      ${dotsSvg}
    </svg>
  `;
}

function renderRocChart(rocData) {
  const container = document.getElementById("rocChartSvg");
  if (!container || !rocData) return;

  const w = 460;
  const h = 220;
  const padL = 45;
  const padR = 25;
  const padT = 20;
  const padB = 35;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const getX = (fpr) => padL + fpr * plotW;
  const getY = (tpr) => padT + plotH - tpr * plotH;

  // Grid lines
  let gridSvg = "";
  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach((v) => {
    const y = getY(v);
    const x = getX(v);
    gridSvg += `
      <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="2,3"/>
      <line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="2,3"/>
      <text x="${padL - 8}" y="${y + 4}" fill="var(--wa-text-secondary)" font-size="10" text-anchor="end" font-family="monospace">${v.toFixed(1)}</text>
      <text x="${x}" y="${h - 14}" fill="var(--wa-text-secondary)" font-size="10" text-anchor="middle" font-family="monospace">${v.toFixed(1)}</text>
    `;
  });

  const pts = rocData.points || [
    { fpr: 0.0, tpr: 0.0 },
    { fpr: 0.0, tpr: 0.0417 },
    { fpr: 0.0, tpr: 0.9583 },
    { fpr: 0.1667, tpr: 0.9583 },
    { fpr: 0.1667, tpr: 1.0 },
    { fpr: 1.0, tpr: 1.0 }
  ];

  const polyPts = pts.map((p) => `${getX(p.fpr).toFixed(1)},${getY(p.tpr).toFixed(1)}`).join(" ");
  const areaPts = `${getX(0)},${getY(0)} ` + polyPts + ` ${getX(1)},${getY(0)}`;

  const opX = getX(rocData.operating_fpr || 0.0417);
  const opY = getY(rocData.operating_tpr || 0.9583);

  container.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}">
      <defs>
        <linearGradient id="rocGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2563eb" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="#2563eb" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      ${gridSvg}
      <polygon fill="url(#rocGrad)" points="${areaPts}"/>
      <line x1="${getX(0)}" y1="${getY(0)}" x2="${getX(1)}" y2="${getY(1)}" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="4,4"/>
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="var(--wa-border)" stroke-width="1.5"/>
      <line x1="${padL}" y1="${padT + plotH}" x2="${w - padR}" y2="${padT + plotH}" stroke="var(--wa-border)" stroke-width="1.5"/>
      <polyline fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${polyPts}"/>
      <circle cx="${opX}" cy="${opY}" r="6" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>
      <text x="${opX + 10}" y="${opY - 6}" fill="#ef4444" font-size="11" font-weight="700">Operating Point (Thresh 0.50)</text>
      <text x="${w / 2 + 10}" y="${h - 2}" fill="var(--wa-text-secondary)" font-size="11" text-anchor="middle">False Positive Rate (1 - Specificity)</text>
      <text x="12" y="${h / 2}" fill="var(--wa-text-secondary)" font-size="11" text-anchor="middle" transform="rotate(-90 12,${h / 2})">True Positive Rate (Recall)</text>
    </svg>
  `;
}

function renderEpochTable(history) {
  const tbody = document.getElementById("epochTableBody");
  if (!tbody || !history) return;

  tbody.innerHTML = history.map((h) => {
    let stateBadge = `<span class="epoch-badge learning">Learning</span>`;
    if (h.epoch <= 3) {
      stateBadge = `<span class="epoch-badge" style="background:rgba(245,158,11,0.15);color:#f59e0b">Warmup</span>`;
    } else if (h.epoch >= 20) {
      stateBadge = `<span class="epoch-badge converged">Converged</span>`;
    } else if (h.epoch >= 15) {
      stateBadge = `<span class="epoch-badge" style="background:rgba(168,85,247,0.15);color:#a855f7">Fine-Tuning</span>`;
    }

    return `
      <tr>
        <td><strong>Epoch ${h.epoch}</strong></td>
        <td style="color:#ef4444">${h.loss.toFixed(4)}</td>
        <td style="color:#f97316">${h.val_loss.toFixed(4)}</td>
        <td style="color:#10b981;font-weight:600">${h.train_accuracy.toFixed(2)}%</td>
        <td style="color:#06b6d4;font-weight:600">${h.validation_accuracy.toFixed(2)}%</td>
        <td style="color:#8b5cf6;font-weight:600">${h.testing_accuracy.toFixed(2)}%</td>
        <td>${stateBadge}</td>
      </tr>
    `;
  }).join("");
}

function renderAllVivaVisualizations() {
  if (MODEL_METRICS && MODEL_METRICS.epoch_history) {
    renderAccuracyChart(MODEL_METRICS.epoch_history);
    renderLossChart(MODEL_METRICS.epoch_history);
    renderEpochTable(MODEL_METRICS.epoch_history);
  }
  if (MODEL_METRICS && MODEL_METRICS.roc_curve) {
    renderRocChart(MODEL_METRICS.roc_curve);
  }
}

function initVivaTabs() {
  const tabBtns = document.querySelectorAll(".viva-tab-btn");
  const tabPanes = document.querySelectorAll(".viva-tab-pane");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add("active");

      if (targetId === "tab-curves" || targetId === "tab-roc") {
        setTimeout(renderAllVivaVisualizations, 50);
      }
    });
  });
}

async function syncLiveMetrics() {
  try {
    const res = await fetch("http://127.0.0.1:5000/metrics");
    if (res.ok) {
      const data = await res.json();
      Object.assign(MODEL_METRICS, data);
      renderAllVivaVisualizations();
    }
  } catch (e) {
    // Falls back to static MODEL_METRICS
  }
}

function openMetricsModal() {
  metricsModal.classList.add("open");
  headerDropdown.classList.remove("open");
  renderAllVivaVisualizations();
}

closeMetricsBtn.addEventListener("click", () => metricsModal.classList.remove("open"));
metricsModal.addEventListener("click", (e) => {
  if (e.target === metricsModal) metricsModal.classList.remove("open");
});
openMetricsAction.addEventListener("click", openMetricsModal);
statusPill.addEventListener("click", openMetricsModal);

// Initialize Viva Tabs & Initial Visualizations
initVivaTabs();
renderAllVivaVisualizations();
syncLiveMetrics();

// =========================================================================
// Event Listeners & Interactions
// =========================================================================

// Text Input typing toggles Mic / Send icon
textInput.addEventListener("input", () => {
  const hasText = textInput.value.trim().length > 0;
  if (hasText) {
    sendBtn.classList.add("send-active");
    sendBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
        <path d="M1.101 21.757L23.8 12.028 1.101 2.3l.011 7.912 13.623 1.816-13.623 1.817-.011 7.912z"/>
      </svg>
    `;
    sendBtn.title = "Send message";
  } else {
    sendBtn.classList.remove("send-active");
    sendBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
        <path d="M11.999 14.942c2.001 0 3.627-1.625 3.627-3.627V5.627C15.626 3.625 14 2 11.999 2 9.998 2 8.372 3.625 8.372 5.627v5.688c0 2.002 1.626 3.627 3.627 3.627zm5.308-3.627c0 2.928-2.38 5.308-5.308 5.308s-5.308-2.38-5.308-5.308H5.163c0 3.493 2.607 6.381 6.012 6.883v3.084h1.649v-3.084c3.405-.502 6.012-3.39 6.012-6.883h-1.529z"/>
      </svg>
    `;
    sendBtn.title = "Voice memo";
  }
});

// Send on Enter Key
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

function sendMessage() {
  const text = textInput.value.trim();
  if (!text) return;
  const newMsg = { type: "text", direction: currentSender, text: text, time: timeNow() };
  if (!CHAT_DATA[currentChatId].initialMessages) CHAT_DATA[currentChatId].initialMessages = [];
  const newIdx = CHAT_DATA[currentChatId].initialMessages.length;
  CHAT_DATA[currentChatId].initialMessages.push(newMsg);
  localStorage.setItem("wa_msgs_" + currentChatId, JSON.stringify(CHAT_DATA[currentChatId].initialMessages));

  appendTextMessage(text, currentSender, newMsg.time, newIdx);
  textInput.value = "";
  textInput.dispatchEvent(new Event("input"));
}

// Attachment Menu toggle
attachBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  attachmentMenu.classList.toggle("open");
  emojiPopover.classList.remove("open");
});

document.addEventListener("click", () => {
  attachmentMenu.classList.remove("open");
  emojiPopover.classList.remove("open");
  headerDropdown.classList.remove("open");
});

attachmentMenu.addEventListener("click", (e) => e.stopPropagation());

// File input trigger
document.getElementById("attachPhotos").addEventListener("click", () => {
  attachmentMenu.classList.remove("open");
  fileInput.click();
});

document.getElementById("attachDoc").addEventListener("click", () => {
  attachmentMenu.classList.remove("open");
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  [...fileInput.files].forEach((f) => appendImageMessage(f, "out"));
  fileInput.value = "";
});

// Quick Demo Stego and Clean Samples
document.getElementById("attachStegoDemo").addEventListener("click", () => {
  attachmentMenu.classList.remove("open");
  if (typeof SAMPLE_IMAGES !== "undefined" && SAMPLE_IMAGES.stego1) {
    appendImageMessage(SAMPLE_IMAGES.stego1, "out");
  }
});

document.getElementById("attachCleanDemo").addEventListener("click", () => {
  attachmentMenu.classList.remove("open");
  if (typeof SAMPLE_IMAGES !== "undefined" && SAMPLE_IMAGES.clean1) {
    appendImageMessage(SAMPLE_IMAGES.clean1, "out");
  }
});

// Drag & Drop Handling
const chatContainer = document.getElementById("chatContainer");

["dragenter", "dragover"].forEach((evt) => {
  chatContainer.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropOverlay.classList.add("active");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropOverlay.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropOverlay.classList.remove("active");
  });
});

dropOverlay.addEventListener("drop", (e) => {
  const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
  files.forEach((f) => appendImageMessage(f, "out"));
});

// Emoji Popover
emojiBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  emojiPopover.classList.toggle("open");
  attachmentMenu.classList.remove("open");
});

document.querySelectorAll(".emoji-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    textInput.value += btn.textContent;
    textInput.focus();
    textInput.dispatchEvent(new Event("input"));
    emojiPopover.classList.remove("open");
  });
});

// Contact List Switching
document.querySelectorAll(".chat-item").forEach((item) => {
  item.addEventListener("click", () => switchChat(item.dataset.chat));
});

// Theme Switcher
themeToggleBtn.addEventListener("click", () => {
  const currentTheme = document.body.getAttribute("data-theme") || "dark";
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  document.body.setAttribute("data-theme", newTheme);
  themeToggleBtn.title = newTheme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
});

// View Mode Toggle (WhatsApp Desktop Web vs Phone Frame)
viewModeBtn.addEventListener("click", () => {
  const waWindow = document.getElementById("waWindow");
  waWindow.classList.toggle("mode-phone");
  const isPhone = waWindow.classList.contains("mode-phone");
  viewModeBtn.classList.toggle("active", isPhone);
  viewModeBtn.title = isPhone ? "Switch to Full WhatsApp Web" : "Switch to Phone Frame Mockup";
});

// 3-Dots Menu Dropdown
menuToggleBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  headerDropdown.classList.toggle("open");
});

// Function to update a contact's name and persist it
function updateContactName(chatId, newName) {
  if (!newName || !newName.trim()) return;
  newName = newName.trim();
  CHAT_DATA[chatId].name = newName;
  const initials = getInitials(newName);
  CHAT_DATA[chatId].avatarText = initials;
  localStorage.setItem("wa_contact_name_" + chatId, newName);

  if (chatId === currentChatId) {
    document.getElementById("chatHeaderName").textContent = newName;
    document.getElementById("chatHeaderAvatar").textContent = initials;
  }

  const sideItem = document.querySelector(`.chat-item[data-chat="${chatId}"]`);
  if (sideItem) {
    const sideName = sideItem.querySelector(".chat-name");
    if (sideName) sideName.textContent = newName;
    const sideAvatar = sideItem.querySelector(".chat-avatar");
    if (sideAvatar) {
      const ind = sideAvatar.querySelector(".online-indicator");
      sideAvatar.textContent = initials;
      if (ind) sideAvatar.appendChild(ind);
    }
  }
}

// Edit Contact Name Button Handler
if (editContactBtn) {
  editContactBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const currentName = CHAT_DATA[currentChatId].name;
    const newName = prompt("Enter contact name (e.g. your friend's name):", currentName);
    if (newName !== null && newName.trim() !== "") {
      updateContactName(currentChatId, newName.trim());
    }
  });
}

// Sender Switcher Toggle (You / Friend)
if (senderToggleBtn) {
  senderToggleBtn.addEventListener("click", () => {
    if (currentSender === "out") {
      currentSender = "in";
      senderToggleBtn.className = "sender-toggle-btn mode-in";
      senderToggleIcon.textContent = "👥";
      senderToggleText.textContent = "Friend";
      senderToggleBtn.title = "Current sender: Friend (Incoming). Click to switch to You (Outgoing)";
    } else {
      currentSender = "out";
      senderToggleBtn.className = "sender-toggle-btn mode-out";
      senderToggleIcon.textContent = "👤";
      senderToggleText.textContent = "You";
      senderToggleBtn.title = "Current sender: You (Outgoing). Click to switch to Friend (Incoming)";
    }
  });
}

const clearChatAction = document.getElementById("clearChatAction");
if (clearChatAction) {
  clearChatAction.addEventListener("click", () => {
    headerDropdown.classList.remove("open");
    localStorage.removeItem("wa_msgs_" + currentChatId);
    switchChat(currentChatId);
  });
}

// Initialize first chat
switchChat("alex");
