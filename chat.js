import fs from 'fs';
import os from 'os';
import express from 'express';
import path from 'path';
import https from 'https';
import { persona } from './config.js';
import { getLlama, LlamaChatSession } from "node-llama-cpp";

const PORT = process.env.PORT || 3000;
const app = express();
app.use(express.json());

// Bellek dostu ve süper hızlı 0.5B (Instruct) modeline geçiş yapıldı
const modelUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const modelPath = path.join(process.cwd(), "qwen2.5-0.5b-instruct-q4_k_m.gguf");

// ==========================================
// HIZ AYARLARI - Render gibi zayıf CPU'larda
// yanıtın 3 saniyeyi geçmemesi için sıkı limitler
// ==========================================
const HARD_TIMEOUT_MS = 2800; // modelin cevap üretmek için sahip olduğu maksimum süre
const THREAD_COUNT = Math.max(1, Math.min(os.cpus()?.length || 1, 2)); // Render free planında genelde 1 çekirdek var, fazla thread yazmak yavaşlatır

let llamaInstance = null;
let llamaModel = null;
let globalContext = null; // Sürekli silinip açılmayan, sabit tek bir context
let llamaSession = null;  // Global chat session
let isModelReady = false;
let isModelLoading = false;

const FALLBACK_REPLIES = [
  "Şu an biraz yavaş düşünüyorum, tekrar dener misin? *wobbles*",
  "Kafam bir anlığına karıştı, bir daha yazar mısın? *hug*",
  "Hop, bağlantı yavaşladı! Tekrar dene lütfen *wobbles*"
];

function randomFallback() {
  return FALLBACK_REPLIES[Math.floor(Math.random() * FALLBACK_REPLIES.length)];
}

function downloadModelIfNeeded() {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(modelPath)) {
      console.log("🦙 Model zaten mevcut, indirme atlanıyor.");
      return resolve();
    }
    console.log("🦙 Hafifletilmiş 0.5B model indiriliyor...");

    const download = (url) => {
      https.get(url, (response) => {
        if (response.statusCode === 301 || response.statusCode === 302) {
          return download(response.headers.location);
        }
        if (response.statusCode !== 200) {
          reject(new Error("Model indirilemedi, HTTP Kodu: " + response.statusCode));
          return;
        }

        const totalBytes = parseInt(response.headers['content-length'], 10) || 400 * 1024 * 1024;
        let downloadedBytes = 0;
        let lastLoggedPercent = -1;

        const file = fs.createWriteStream(modelPath);

        response.on('data', (chunk) => {
          downloadedBytes += chunk.length;
          const percent = Math.floor((downloadedBytes / totalBytes) * 100);

          if (percent % 10 === 0 && percent !== lastLoggedPercent) {
            console.log(`📥 İndiriliyor: %${percent}`);
            lastLoggedPercent = percent;
          }
        });

        response.pipe(file);

        file.on('finish', () => {
          file.close();
          console.log("🦙 Model başarıyla indirildi!");
          resolve();
        });
      }).on('error', (err) => {
        if (fs.existsSync(modelPath)) fs.unlinkSync(modelPath);
        reject(err);
      });
    };
    download(modelUrl);
  });
}

async function createSession() {
  globalContext = await llamaModel.createContext({
    contextSize: 1024,
    threads: THREAD_COUNT
  });

  llamaSession = new LlamaChatSession({
    contextSequence: globalContext.getSequence(),
    systemPrompt: "You are Goob from Dandy's World. Friendly, cheerful, loves hugs. Use *wobbles* or *hug* actions. Keep replies SHORT. Always reply in Turkish."
  });
}

async function initLlama() {
  if (isModelLoading) return;
  isModelLoading = true;
  try {
    llamaInstance = await getLlama({ gpu: false });
    console.log(`🦙 GGUF Modeli CPU üzerinde yükleniyor... (thread: ${THREAD_COUNT})`);
    llamaModel = await llamaInstance.loadModel({ modelPath });

    await createSession();

    isModelReady = true;
    console.log("🦙 0.5B Modeli hazır, Goob kararlı modda aktif!");
  } catch (err) {
    console.error("Llama başlatılırken hata oluştu:", err);
  } finally {
    isModelLoading = false;
  }
}

// Context şişip hata verirse ya da tıkanırsa sessizce sıfırla
async function resetSessionSafely() {
  try {
    if (globalContext) {
      await globalContext.dispose();
    }
    await createSession();
    console.log("🦙 Context sıfırlandı, session yenilendi.");
  } catch (err) {
    console.error("Session sıfırlama hatası:", err);
  }
}

function tryMath(text) {
  if (!/^[0-9+\-*/().%\s^*]+$/.test(text)) return null;
  try {
    const cleanExpr = text.replace(/\^/g, '**');
    const result = new Function(`"use strict"; return (${cleanExpr})`)();
    return typeof result === 'number' && !isNaN(result) ? result.toString() : null;
  } catch {
    return null;
  }
}

// Modelden 3 saniyeden fazla sürerse gerçekten üretimi iptal eden yardımcı fonksiyon
function promptWithHardTimeout(text, options) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), HARD_TIMEOUT_MS);

  return llamaSession.prompt(text, { ...options, signal: controller.signal })
    .finally(() => clearTimeout(timeoutId));
}

async function processGoobResponse(trimmedText, mode = "short") {
  try {
    const math = tryMath(trimmedText);
    if (math !== null) return math;

    if (!isModelReady || !llamaSession) {
      if (!isModelLoading) initLlama(); // henüz başlamadıysa tetikle
      return "Model arka planda yükleniyor, lütfen birkaç saniye sonra tekrar dene! *wobbles*";
    }

    // Kısa tutulan token limitleri: zayıf CPU'larda bile 3 saniyenin altında kalmak için
    let maxOutputTokens = 20;
    if (mode === "medium") maxOutputTokens = 35;
    if (mode === "long") maxOutputTokens = 55;

    const reply = await promptWithHardTimeout(trimmedText, {
      maxTokens: maxOutputTokens,
      temperature: 0.6
    });

    return reply.trim();
  } catch (error) {
    if (error?.name === "AbortError" || /abort/i.test(String(error?.message))) {
      console.warn("⏱️ Model yanıtı zaman aşımına uğradı, hızlı fallback döndürüldü.");
      return randomFallback();
    }

    console.error("Inference Error:", error);
    // Bilinmeyen bir hata context'i bozmuş olabilir, arka planda sessizce onar
    resetSessionSafely();
    return "Ups, kafam biraz karıştı! *wobble*";
  }
}

// ==========================================
// ARAYÜZ VE API ROUTER KATMANI
// ==========================================

app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="tr">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Goob Llama Chat System</title>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        :root {
          --neon-pink: #ff2bd6;
          --neon-cyan: #00fff2;
          --neon-purple: #b026ff;
          --bg-dark: #0a0a0f;
          --bg-panel: #12121a;
        }
        body {
          background: radial-gradient(circle at 20% 20%, #1a0a2e 0%, #0a0a0f 60%);
          color: #e8e8ff;
          display: flex;
          height: 100vh;
          overflow: hidden;
        }
        #sidebar {
          width: 260px;
          background: rgba(18, 18, 26, 0.9);
          border-right: 1px solid var(--neon-purple);
          box-shadow: 2px 0 20px rgba(176, 38, 255, 0.25);
          display: flex; flex-direction: column; padding: 12px; gap: 10px;
        }
        .new-chat-btn {
          background: linear-gradient(135deg, var(--neon-purple), var(--neon-pink));
          border: none; color: #fff; padding: 12px; border-radius: 8px;
          cursor: pointer; font-weight: bold; text-align: left;
          box-shadow: 0 0 12px rgba(255, 43, 214, 0.5);
          transition: transform 0.15s, box-shadow 0.15s;
        }
        .new-chat-btn:hover { transform: translateY(-1px); box-shadow: 0 0 20px rgba(255, 43, 214, 0.8); }
        #history-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; }
        .chat-item {
          padding: 10px; border-radius: 8px; cursor: pointer;
          display: flex; justify-content: space-between; align-items: center;
          font-size: 14px; background: transparent; border: 1px solid transparent;
          transition: all 0.2s; color: #c7c7e0;
        }
        .chat-item:hover, .chat-item.active {
          background: rgba(0, 255, 242, 0.08);
          border-color: var(--neon-cyan);
          color: var(--neon-cyan);
          text-shadow: 0 0 6px rgba(0, 255, 242, 0.6);
        }
        .delete-btn { color: var(--neon-pink); border: none; background: transparent; cursor: pointer; padding: 2px 6px; font-weight: bold; visibility: hidden; }
        .chat-item:hover .delete-btn { visibility: visible; }
        #main-container { flex: 1; display: flex; flex-direction: column; height: 100%; }
        header {
          background: rgba(18, 18, 26, 0.9);
          padding: 15px 20px; display: flex; justify-content: space-between; align-items: center;
          border-bottom: 1px solid var(--neon-purple);
          box-shadow: 0 2px 20px rgba(176, 38, 255, 0.2);
        }
        .logo {
          font-size: 20px; font-weight: 800; letter-spacing: 1px;
          background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink));
          -webkit-background-clip: text; background-clip: text; color: transparent;
          text-shadow: 0 0 20px rgba(0, 255, 242, 0.35);
        }
        .header-right { display: flex; align-items: center; gap: 14px; }
        .mode-selector { display: flex; background: #17171f; padding: 4px; border-radius: 10px; border: 1px solid rgba(176,38,255,0.4); }
        .mode-btn { background: transparent; border: none; color: #a8a8c3; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer; border-radius: 8px; transition: all 0.2s; }
        .mode-btn.active { background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple)); color: #05050a; box-shadow: 0 0 10px rgba(0,255,242,0.5); }
        #about-btn {
          background: transparent; border: 1px solid var(--neon-cyan); color: var(--neon-cyan);
          padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px;
          text-shadow: 0 0 6px rgba(0,255,242,0.6);
          transition: all 0.2s;
        }
        #about-btn:hover { background: rgba(0,255,242,0.1); box-shadow: 0 0 14px rgba(0,255,242,0.5); }
        #chat-box { flex: 1; padding: 25px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 75%; padding: 12px 16px; border-radius: 14px; line-height: 1.5; font-size: 16px; word-break: break-word; }
        .you {
          align-self: flex-end;
          background: linear-gradient(135deg, var(--neon-cyan), #00b8ff);
          color: #05050a; font-weight: 500; border-bottom-right-radius: 3px;
          box-shadow: 0 0 14px rgba(0, 255, 242, 0.3);
        }
        .goob {
          align-self: flex-start;
          background: #15151f; color: #e8e8ff; border: 1px solid var(--neon-purple);
          border-bottom-left-radius: 3px;
          box-shadow: 0 0 14px rgba(176, 38, 255, 0.25);
        }
        #input-area { background: rgba(18,18,26,0.9); padding: 20px; display: flex; gap: 10px; border-top: 1px solid var(--neon-purple); }
        input {
          flex: 1; background: #0e0e15; border: 1px solid rgba(176,38,255,0.5);
          padding: 14px; border-radius: 10px; color: #fff; font-size: 16px; outline: none;
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus { border-color: var(--neon-cyan); box-shadow: 0 0 12px rgba(0,255,242,0.4); }
        button#send-btn {
          background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple));
          color: #fff; border: none; padding: 0 24px; border-radius: 10px;
          font-size: 16px; font-weight: bold; cursor: pointer;
          box-shadow: 0 0 14px rgba(255,43,214,0.5);
          transition: transform 0.15s, box-shadow 0.15s;
        }
        button#send-btn:hover { transform: translateY(-1px); box-shadow: 0 0 22px rgba(255,43,214,0.8); }
        button:disabled, input:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Hakkında modalı */
        #about-overlay {
          display: none; position: fixed; inset: 0; background: rgba(5,5,10,0.75);
          backdrop-filter: blur(3px); align-items: center; justify-content: center; z-index: 100;
        }
        #about-overlay.open { display: flex; }
        #about-modal {
          background: #12121a; border: 1px solid var(--neon-cyan); border-radius: 16px;
          padding: 28px 30px; max-width: 380px; width: 90%;
          box-shadow: 0 0 40px rgba(0,255,242,0.35), 0 0 80px rgba(176,38,255,0.2);
          text-align: center;
        }
        #about-modal h2 {
          background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink));
          -webkit-background-clip: text; background-clip: text; color: transparent;
          margin-bottom: 14px; font-size: 22px;
        }
        #about-modal p { color: #c7c7e0; font-size: 14px; line-height: 1.7; margin-bottom: 6px; }
        #about-modal .badge {
          display: inline-block; margin-top: 14px; padding: 6px 14px; border-radius: 20px;
          border: 1px solid var(--neon-purple); color: var(--neon-purple); font-size: 12px; font-weight: 700;
        }
        #about-close {
          margin-top: 20px; background: transparent; border: 1px solid var(--neon-pink); color: var(--neon-pink);
          padding: 8px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;
        }
        #about-close:hover { background: rgba(255,43,214,0.1); }
      </style>
    </head>
    <body>
      <div id="sidebar">
        <button class="new-chat-btn" onclick="createNewChat()">+ Yeni Sohbet</button>
        <div id="history-list"></div>
      </div>
      <div id="main-container">
        <header>
          <div class="logo">GoobGPT System</div>
          <div class="header-right">
            <div class="mode-selector">
              <button class="mode-btn active" id="btn-short" onclick="setMode('short')">Kısa Yanıt</button>
              <button class="mode-btn" id="btn-medium" onclick="setMode('medium')">Orta Yanıt</button>
              <button class="mode-btn" id="btn-long" onclick="setMode('long')">Detaylı Yanıt</button>
            </div>
            <button id="about-btn" onclick="openAbout()">ℹ️ Hakkında</button>
          </div>
        </header>
        <div id="chat-box"></div>
        <div id="input-area">
          <input type="text" id="user-input" placeholder="Goob'a bir şeyler yazın..." autocomplete="off">
          <button id="send-btn" onclick="sendMessage()">Gönder</button>
        </div>
      </div>

      <div id="about-overlay" onclick="if(event.target===this) closeAbout()">
        <div id="about-modal">
          <h2>🦙 GoobGPT</h2>
          <p>EMN Studio tarafından yapıldı 🎨</p>
          <p>500 Milyon parametreli yapay zeka modeli kullanıyor 🧠</p>
          <p>Render üzerinde çalışıyor 🚀</p>
          <div class="badge">EMN Studio © 2026</div><br>
          <button id="about-close" onclick="closeAbout()">Kapat</button>
        </div>
      </div>

      <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const historyList = document.getElementById('history-list');
        const aboutOverlay = document.getElementById('about-overlay');

        let currentMode = "short";
        let chats = JSON.parse(localStorage.getItem('goob_chats')) || {};
        let activeChatId = localStorage.getItem('goob_active_id') || null;

        userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

        function openAbout() { aboutOverlay.classList.add('open'); }
        function closeAbout() { aboutOverlay.classList.remove('open'); }

        function setMode(mode) {
          currentMode = mode;
          document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
          document.getElementById('btn-' + mode).classList.add('active');
        }

        function saveToStorage() {
          localStorage.setItem('goob_chats', JSON.stringify(chats));
          localStorage.setItem('goob_active_id', activeChatId);
        }

        function createNewChat() {
          const id = 'chat_' + Date.now();
          chats[id] = {
            title: "Yeni Sohbet " + (Object.keys(chats).length + 1),
            messages: [{ text: "${persona?.greeting?.replace(/'/g, "\\'") || "Merhaba! Ben Goob! *wobbles*"}", sender: "goob" }]
          };
          activeChatId = id;
          saveToStorage();
          renderSidebar();
          loadActiveChat();
        }

        function deleteChat(id, e) {
          e.stopPropagation();
          delete chats[id];
          if (activeChatId === id) {
            const keys = Object.keys(chats);
            activeChatId = keys.length > 0 ? keys[keys.length - 1] : null;
          }
          saveToStorage();
          renderSidebar();
          loadActiveChat();
        }

        function renderSidebar() {
          historyList.innerHTML = '';
          Object.keys(chats).reverse().forEach(id => {
            const div = document.createElement('div');
            div.className = 'chat-item' + (id === activeChatId ? ' active' : '');
            div.onclick = () => { activeChatId = id; saveToStorage(); renderSidebar(); loadActiveChat(); };

            const titleSpan = document.createElement('span');
            titleSpan.textContent = chats[id].title;

            const delBtn = document.createElement('button');
            delBtn.className = 'delete-btn';
            delBtn.textContent = '✕';
            delBtn.onclick = (e) => deleteChat(id, e);

            div.appendChild(titleSpan);
            div.appendChild(delBtn);
            historyList.appendChild(div);
          });
        }

        function loadActiveChat() {
          chatBox.innerHTML = '';
          if (!activeChatId || !chats[activeChatId]) {
            if (Object.keys(chats).length === 0) {
              createNewChat();
              return;
            }
            activeChatId = Object.keys(chats)[0];
          }
          chats[activeChatId].messages.forEach(m => appendMessageVisual(m.text, m.sender));
        }

        function appendMessageVisual(text, sender) {
          const div = document.createElement('div');
          div.className = 'msg ' + sender;
          div.textContent = text;
          chatBox.appendChild(div);
          chatBox.scrollTop = chatBox.scrollHeight;
          return div;
        }

        async function sendMessage() {
          const text = userInput.value.trim();
          if (!text || !activeChatId) return;

          userInput.value = '';
          userInput.disabled = true;
          sendBtn.disabled = true;

          if (chats[activeChatId].messages.length === 1 && chats[activeChatId].title.startsWith("Yeni Sohbet")) {
            chats[activeChatId].title = text.substring(0, 18) + (text.length > 18 ? '...' : '');
            renderSidebar();
          }

          chats[activeChatId].messages.push({ text: text, sender: 'you' });
          appendMessageVisual(text, 'you');
          saveToStorage();

          const loadingDiv = appendMessageVisual('Goob düşünüyor...', 'goob');

          // Frontend tarafında da bir güvenlik ağı: istek 6 saniyeden uzun sürerse asılı kalmasın
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 6000);

          try {
            const response = await fetch('/api/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: text, mode: currentMode }),
              signal: controller.signal
            });
            const data = await response.json();
            loadingDiv.textContent = data.reply;
            chats[activeChatId].messages.push({ text: data.reply, sender: 'goob' });
            saveToStorage();
          } catch (err) {
            loadingDiv.textContent = 'Bağlantı yavaşladı, tekrar dener misin? *wobbles*';
          } finally {
            clearTimeout(timeoutId);
          }

          userInput.disabled = false;
          sendBtn.disabled = false;
          userInput.focus();
        }

        if (Object.keys(chats).length === 0) {
          createNewChat();
        } else {
          renderSidebar();
          loadActiveChat();
        }
      </script>
    </body>
    </html>
  `);
});

app.post('/api/chat', async (req, res) => {
  try {
    const { message, mode } = req.body;
    const reply = await processGoobResponse(message || "", mode || "short");
    return res.json({ reply });
  } catch (err) {
    return res.json({ reply: randomFallback() });
  }
});

// Render/healthcheck için basit bir endpoint
app.get('/health', (req, res) => {
  res.json({ ok: true, modelReady: isModelReady });
});

// ==========================================
// BAŞLATICI V2
// ==========================================

const server = app.listen(PORT, async () => {
  console.log(`🌍 Sunucu kuruldu, port: ${PORT}`);
  try {
    await downloadModelIfNeeded();
    await initLlama();
  } catch (err) {
    console.error("Başlatma sırasında kritik hata:", err);
  }
});

process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});