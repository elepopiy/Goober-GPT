import fs from 'fs';
import express from 'express';
import path from 'path';
import http from 'https';
import { persona } from './config.js';
import { getLlama, LlamaChatSession } from "node-llama-cpp";

const PORT = process.env.PORT || 3000;
const app = express();
app.use(express.json());

// Tamamen halka açık, token istemeyen ve 512MB RAM dostu 0.5B modeli
const modelUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const modelPath = path.join(process.cwd(), "qwen2.5-0.5b-instruct-q4_k_m.gguf");

let llamaSession = null;
let isModelReady = false;

function downloadModelIfNeeded() {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(modelPath)) {
      console.log("🦙 Model zaten mevcut, indirme atlanıyor.");
      return resolve();
    }
    console.log("🦙 Hafif 0.5B model indiriliyor... İlerleme takibi başlatıldı.");

    const download = (url) => {
      http.get(url, (response) => {
        if (response.statusCode === 301 || response.statusCode === 302) {
          return download(response.headers.location);
        }
        if (response.statusCode !== 200) {
          reject(new Error("Model indirilemedi, HTTP Kodu: " + response.statusCode));
          return;
        }

        const totalBytes = parseInt(response.headers['content-length'], 10) || 380 * 1024 * 1024;
        let downloadedBytes = 0;
        let lastLoggedPercent = -1;

        const file = fs.createWriteStream(modelPath);
        
        response.on('data', (chunk) => {
          downloadedBytes += chunk.length;
          const percent = Math.floor((downloadedBytes / totalBytes) * 100);
          
          // Her %5 artışta bir log basarak paneli rahatlatıyoruz
          if (percent % 5 === 0 && percent !== lastLoggedPercent) {
            const downloadedMB = (downloadedBytes / (1024 * 1024)).toFixed(1);
            const totalMB = (totalBytes / (1024 * 1024)).toFixed(1);
            console.log(`📥 İndiriliyor: %${percent} (${downloadedMB} MB / ${totalMB} MB)`);
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

async function initLlama() {
  try {
    const llama = await getLlama({ gpu: false });
    console.log("🦙 GGUF Modeli CPU üzerinde yükleniyor...");
    const llamaModel = await llama.loadModel({ modelPath });
    
    const context = await llamaModel.createContext({
      contextSize: 512, 
      threads: 4 
    });
    
    llamaSession = new LlamaChatSession({
      contextSequence: context.getSequence(),
      systemPrompt: `You are Goob from Dandy's World. Friendly, cheerful, loves hugs. Use *wobbles* or *hug* actions. Reply in Turkish or English based on the user.`
    });
    isModelReady = true;
    console.log("🦙 GGUF Modeli tamamen hazır!");
  } catch (err) {
    console.error("Llama başlatılırken hata oluştu:", err);
  }
}

function tryMath(text) {
  if (!/^[0-9+\-*/().%\s^*]+$/.test(text)) return null;
  try {
    const expr = text.replace(/\^/g, '**');
    return Function(`"use strict"; return (${expr})`)().toString();
  } catch {
    return null;
  }
}

async function processGoobResponse(trimmedText, mode = "short") {
  try {
    const math = tryMath(trimmedText);
    if (math !== null) return math;

    if (!isModelReady || !llamaSession) return "Model arka planda yükleniyor, lütfen birkaç saniye sonra tekrar deneyin! *wobbles*";

    let maxOutputTokens = 25; 
    let temperature = 0.4; 

    if (mode === "medium") {
      maxOutputTokens = 60;
      temperature = 0.6;
    } else if (mode === "long") {
      maxOutputTokens = 130;
      temperature = 0.8;
    }

    const reply = await llamaSession.prompt(trimmedText, {
      maxOutputTokens: maxOutputTokens, 
      temperature: temperature,
      stopOnTokens: ["\n", "\n\n", "<END>", "<USER>", "User:", "Goob:"] 
    });

    return reply.trim();
  } catch (error) {
    console.error("Inference Error:", error);
    return "Ups, kafam biraz karıştı! *wobble*";
  }
}

// ==========================================
// 3. WEB SİTESİ ARAYÜZÜ (SIDEBAR & MOD SEÇİCİ)
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
        body { background: #121214; color: #e1e1e6; display: flex; height: 100vh; overflow: hidden; }
        #sidebar { width: 260px; background: #1a1a1e; border-right: 1px solid #29292e; display: flex; flex-direction: column; padding: 10px; gap: 10px; }
        .new-chat-btn { background: #202024; border: 1px solid #29292e; color: #fff; padding: 12px; border-radius: 6px; cursor: pointer; font-weight: bold; text-align: left; transition: background 0.2s; }
        .new-chat-btn:hover { background: #2d2d34; border-color: #00e676; }
        #history-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; }
        .chat-item { padding: 10px; border-radius: 6px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-size: 14px; background: transparent; transition: background 0.2s; }
        .chat-item:hover, .chat-item.active { background: #202024; color: #00e676; }
        .delete-btn { color: #ef5350; border: none; background: transparent; cursor: pointer; padding: 2px 6px; font-weight: bold; visibility: hidden; }
        .chat-item:hover .delete-btn { visibility: visible; }
        #main-container { flex: 1; display: flex; flex-direction: column; height: 100%; background: #121214; }
        header { background: #1a1a1e; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #29292e; }
        .logo { font-size: 18px; font-weight: bold; color: #00e676; }
        .mode-selector { display: flex; background: #202024; padding: 4px; border-radius: 8px; border: 1px solid #29292e; }
        .mode-btn { background: transparent; border: none; color: #a8a8b3; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer; border-radius: 6px; transition: all 0.2s; }
        .mode-btn.active { background: #00e676; color: #0a0a0c; }
        #chat-box { flex: 1; padding: 25px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 16px; word-break: break-word; animation: fadeIn 0.2s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        .you { align-self: flex-end; background: #00e676; color: #0a0a0c; border-bottom-right-radius: 2px; font-weight: 500; }
        .goob { align-self: flex-start; background: #1a1a1e; color: #e1e1e6; border-bottom-left-radius: 2px; border: 1px solid #29292e; }
        #input-area { background: #1a1a1e; padding: 20px; display: flex; gap: 10px; border-top: 1px solid #29292e; }
        input { flex: 1; background: #121214; border: 1px solid #29292e; padding: 14px; border-radius: 8px; color: #fff; font-size: 16px; outline: none; }
        input:focus { border-color: #00e676; }
        button#send-btn { background: #00e676; color: #0a0a0c; border: none; padding: 0 24px; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button#send-btn:hover { background: #00c853; }
        button:disabled, input:disabled { opacity: 0.6; cursor: not-allowed; }
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
          <div class="mode-selector">
            <button class="mode-btn active" id="btn-short" onclick="setMode('short')">Kısa Düşünme</button>
            <button class="mode-btn" id="btn-medium" onclick="setMode('medium')">Orta Düşünme</button>
            <button class="mode-btn" id="btn-long" onclick="setMode('long')">Uzun Düşünme</button>
          </div>
        </header>
        <div id="chat-box"></div>
        <div id="input-area">
          <input type="text" id="user-input" placeholder="Goob'a bir şeyler yazın..." autocomplete="off">
          <button id="send-btn" onclick="sendMessage()">Gönder</button>
        </div>
      </div>
      <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const historyList = document.getElementById('history-list');

        let currentMode = "short";
        let chats = JSON.parse(localStorage.getItem('goob_chats')) || {};
        let activeChatId = localStorage.getItem('goob_active_id') || null;

        userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

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
            messages: [{ text: "${persona.greeting.replace(/'/g, "\\'")}", sender: "goob" }]
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

          try {
            const response = await fetch('/api/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: text, mode: currentMode })
            });
            const data = await response.json();
            loadingDiv.textContent = data.reply;
            chats[activeChatId].messages.push({ text: data.reply, sender: 'goob' });
            saveToStorage();
          } catch (err) {
            loadingDiv.textContent = 'Bir hata oluştu.';
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
    return res.json({ reply: "Hata: " + err.message });
  }
});

// ==========================================
// 4. SUNUCU BAŞLATMA (İLK ÖNCE PORTU AÇIYORUZ)
// ==========================================

const server = app.listen(PORT, async () => {
  console.log(`🌍 Sunucu ${PORT} portunda aktif edildi! Render port kontrolünü geçti.`);
  
  try {
    await downloadModelIfNeeded();
    await initLlama();
  } catch (err) {
    console.error("Arka plan kurulum hatası:", err);
  }
});

process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});