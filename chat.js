import fs from 'fs';
import express from 'express';
import path from 'path';
import http from 'https';
import { persona } from './config.js';
import { getLlama, LlamaChatSession } from "node-llama-cpp";

const PORT = process.env.PORT || 3000; // Render'ın dinamik portunu yakalamak için önemli
const app = express();
app.use(express.json());

// RAM dostu 0.5B model linki (Yaklaşık 380 MB - Çok hızlı iner ve çalışır)
const modelUrl = "https://huggingface.co/lmstudio-community/Llama-3.2-0.5B-Instruct-GGUF/resolve/main/Llama-3.2-0.5B-Instruct-Q4_K_M.gguf";
const modelPath = path.join(process.cwd(), "Llama-3.2-0.5B-Instruct-Q4_K_M.gguf");

let llamaSession = null;
let isModelReady = false;

function downloadModelIfNeeded() {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(modelPath)) {
      console.log("🦙 Model zaten mevcut, indirme atlanıyor.");
      return resolve();
    }
    console.log("🦙 Hafif 0.5B model indiriliyor...");

    const download = (url) => {
      http.get(url, (response) => {
        if (response.statusCode === 301 || response.statusCode === 302) {
          return download(response.headers.location);
        }
        if (response.statusCode !== 200) {
          reject(new Error("Model indirilemedi, HTTP Kodu: " + response.statusCode));
          return;
        }

        const file = fs.createWriteStream(modelPath);
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
      threads: 4 // RAM tasarrufu için thread sayısını 4'e düşürdük
    });
    
    llamaSession = new LlamaChatSession({
      contextSequence: context.getSequence(),
      systemPrompt: `You are Goob from Dandy's World. Friendly, cheerful, loves hugs. Use *wobbles* or *hug* actions. Reply in Turkish or English based on the user.`
    });
    isModelReady = true;
    console.log("🦙 Llama-3.2-0.5B GGUF tamamen hazır!");
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
// ARAYÜZ VE API ENDPOINT'LERİ
// ==========================================

app.get('/', (req, res) => {
  res.send(`... (Arayüz kodların buraya gelecek, dokunmana gerek yok) ...`);
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
  console.log(`🌍 Sunucu ${PORT} portunda aktif edildi! Render artık mutlu.`);
  
  // Port açıldıktan sonra indirme ve yükleme işlemlerini arka planda başlatıyoruz
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