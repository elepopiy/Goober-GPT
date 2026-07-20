// trainer.js — orchestrates the training loop over random chunks of the corpus
import fs from 'fs';
import { Buffer } from 'buffer';
import { crossEntropyLoss } from './loss.js';
import { Adam } from './optimizer.js';

export class Trainer {
  constructor(model, tokenIds, cfg, savePath, rawText = null) {
    this.model = model;
    this.tokenIds = Array.isArray(tokenIds) ? tokenIds : [];
    this.cfg = cfg;
    this.savePath = savePath;
    this.optimizer = new Adam(model.parameters(), cfg.learningRate);

    // <GOOB>\n dizisinin karakter (=token, çünkü char-level) konumlarını bul.
    // Bunlardan sonrası, o örnekte "cevap" bölgesidir.
    this.goobStarts = [];
    if (rawText) {
      const marker = '<GOOB>\n';
      let idx = rawText.indexOf(marker);
      while (idx !== -1) {
        this.goobStarts.push(idx + marker.length);
        idx = rawText.indexOf(marker, idx + 1);
      }
    }
  }

  sampleWindow() {
    const bs = this.cfg.blockSize;
    const maxStart = this.tokenIds.length - bs - 1;
    if (maxStart <= 0) throw new Error('Training text is shorter than blockSize; add more text to data/train.txt');

    if (this.goobStarts.length === 0) {
      // fallback: eski davranış (goobStarts bilgisi yoksa)
      const start = Math.floor(Math.random() * maxStart);
      return { start, mask: null };
    }

    // Rastgele bir <GOOB> cevap başlangıcı seç, pencereyi onun BİTİŞİNE hizala
    // (yani cevap pencerenin sonunda kalsın, mümkün olduğunca soru da içeride olsun)
    const goobStart = this.goobStarts[Math.floor(Math.random() * this.goobStarts.length)];
    let start = Math.min(Math.max(goobStart - bs + Math.floor(bs * 0.3), 0), maxStart);

    const mask = new Array(bs).fill(0);
    for (let i = 0; i < bs; i++) {
      const pos = start + i; // targetIds[i]'nin corpus'taki gerçek pozisyonu = start+1+i, ama basitçe:
      if (pos + 1 >= goobStart) mask[i] = 1; // target token <GOOB> sonrasıysa say
    }
    return { start, mask };
  }

  step() {
    const bs = this.cfg.blockSize;
    const { start, mask } = this.sampleWindow();
    const inputIds = this.tokenIds.slice(start, start + bs);
    const targetIds = this.tokenIds.slice(start + 1, start + bs + 1);

    this.optimizer.zeroGrad();
    const logits = this.model.forward(inputIds);
    const loss = crossEntropyLoss(logits, targetIds, mask);
    loss.backward();
    this.optimizer.step();

    let lVal = loss.data[0];
    if (Number.isNaN(lVal) || !Number.isFinite(lVal)) lVal = 0;
    return lVal;
  }

  train(steps, { logEvery = 20, saveEvery = 200 } = {}) {
    let runningLoss = 0;
    let validSteps = 0;
    for (let i = 1; i <= steps; i++) {
      const lossVal = this.step();
      if (lossVal > 0) {
        runningLoss += lossVal;
        validSteps++;
      }
      if (i % logEvery === 0) {
        const avgLoss = validSteps > 0 ? (runningLoss / validSteps) : 0;
        console.log(`step ${i}/${steps}  loss=${avgLoss.toFixed(4)}`);
        runningLoss = 0;
        validSteps = 0;
      }
      if (this.savePath && i % saveEvery === 0) {
        this.save();
      }
    }
    if (this.savePath) this.save();
  }

  save() {
  try {
    const state = this.model.stateDict();

    const checkpoint = {
      cfg: this.cfg,
      state
    };

    // JSON yerine binary buffer kullan
    const buffer = Buffer.from(
      JSON.stringify(checkpoint, (key, value) => {
        if (value instanceof Float64Array || value instanceof Float32Array) {
          return Array.from(value);
        }
        return value;
      })
    );

    fs.writeFileSync(this.savePath + ".bin", buffer);

    console.log(
      `checkpoint saved -> ${this.savePath}.bin (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`
    );

  } catch (e) {
    console.error(
      `[GOOBER ENGINE CORE ERROR] Failed to save checkpoint:`,
      e
    );
  }
}

}