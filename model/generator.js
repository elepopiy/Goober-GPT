// generator.js — autoregressive sampling (temperature + top-k)
function sampleFromProbs(probs, topK) {
  // GÜÇLENDİRME: Gelen olasılık dizisinde NaN varsa 0'a eşitle
  const cleanProbs = Array.from(probs).map(p => (Number.isNaN(p) || !Number.isFinite(p) ? 0 : p));
  
  const indexed = cleanProbs.map((p, i) => [p, i]);
  indexed.sort((a, b) => b[0] - a[0]);
  
  const top = topK ? indexed.slice(0, topK) : indexed;
  const total = top.reduce((s, [p]) => s + p, 0);
  
  // Eğer tüm olasılıklar sıfır çıktıysa en yüksek ihtimalli ilk elemanı seç
  if (total <= 0) return top[0] ? top[0][1] : 0;

  let r = Math.random() * total;
  for (const [p, i] of top) {
    r -= p;
    if (r <= 0) return i;
  }
  return top[0][1];
}

export function generate(model, tokenizer, prompt, { maxNewTokens = 200, temperature = 0.8, topK = 20, repetitionPenalty = 1.0 } = {}) {
  const cfg = model.cfg;
  let ids = tokenizer.encode(prompt);
  if (!ids || ids.length === 0) ids = [0];

  for (let step = 0; step < maxNewTokens; step++) {
    const context = ids.slice(-cfg.blockSize);
    const logits = model.forward(context); // [seq, vocab]
    
    if (!logits || !logits.data) break;

    const lastRowStart = (context.length - 1) * cfg.vocabSize;
    // Güvenli sınır kontrolü
    if (lastRowStart + cfg.vocabSize > logits.data.length) break;
    
    const lastLogits = logits.data.subarray(lastRowStart, lastRowStart + cfg.vocabSize);

    // GÜÇLENDİRME: Logitlerin içinde NaN/Inf patlaması varsa temizle
    for (let i = 0; i < lastLogits.length; i++) {
      if (Number.isNaN(lastLogits[i])) lastLogits[i] = 0.0;
      if (lastLogits[i] === Infinity) lastLogits[i] = 1e4;
      if (lastLogits[i] === -Infinity) lastLogits[i] = -1e4;
    }

    // Repetition penalty: son üretilen (context içindeki) tokenların logitini küçült.
    // Bu, modelin aynı kelime/öbeği döngüsel şekilde tekrar etmesini azaltır.
    if (repetitionPenalty && repetitionPenalty !== 1.0) {
      const seen = new Set(context);
      for (const tokId of seen) {
        if (tokId < 0 || tokId >= lastLogits.length) continue;
        const val = lastLogits[tokId];
        lastLogits[tokId] = val > 0 ? val / repetitionPenalty : val * repetitionPenalty;
      }
    }

    // temperature-scaled softmax on the last position only
    let maxLogit = -Infinity;
    for (let i = 0; i < lastLogits.length; i++) maxLogit = Math.max(maxLogit, lastLogits[i]);
    
    const exps = new Float64Array(lastLogits.length);
    let sumExp = 0;
    const safeTemp = Math.max(temperature, 1e-6);

    for (let i = 0; i < lastLogits.length; i++) {
      exps[i] = Math.exp((lastLogits[i] - maxLogit) / safeTemp);
      sumExp += exps[i];
    }
    
    // GÜÇLENDİRME: sumExp sıfır veya NaN ise çökme koruması
    if (sumExp <= 0 || Number.isNaN(sumExp)) {
      exps.fill(1.0 / exps.length);
    } else {
      for (let i = 0; i < exps.length; i++) exps[i] /= sumExp;
    }

    const nextId = sampleFromProbs(exps, topK);

    // Eğer üretilen token ID geçersizse döngüyü güvenli bitir
    if (nextId === undefined || Number.isNaN(nextId)) {
      break;
    }

    ids.push(nextId);

    // <END> üretilince üretimi durdur
    const decoded = tokenizer.decode(ids);

    if (decoded.includes("<END>")) {
      break;
    }
  }

  return tokenizer.decode(ids);
}