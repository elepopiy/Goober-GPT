// attention.js — multi-head causal self-attention
import { Module } from './module.js';
import { Linear } from './linear.js';
import { softmaxRows } from './softmax.js';
import { applyCausalMask, sliceCols, concatCols } from './tensor_ops.js';

export class MultiHeadAttention extends Module {
  constructor(dModel, numHeads) {
    super();
    if (dModel % numHeads !== 0) throw new Error('dModel must be divisible by numHeads');
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.headDim = dModel / numHeads;
    
    // Mevcut katman yapıları birebir korundu
    this.Wq = new Linear(dModel, dModel, false);
    this.Wk = new Linear(dModel, dModel, false);
    this.Wv = new Linear(dModel, dModel, false);
    this.Wo = new Linear(dModel, dModel, false);
  }

  forward(x) {
    // x veya x.data null/tanımsız ise koruma sağla
    if (!x || !x.matmul) return x;

    const q = this.Wq.forward(x); // [seq, dModel]
    const k = this.Wk.forward(x);
    const v = this.Wv.forward(x);
    const scale = 1 / Math.sqrt(this.headDim);

    const headOutputs = [];
    for (let h = 0; h < this.numHeads; h++) {
      const start = h * this.headDim;
      const end = start + this.headDim;
      
      const qh = sliceCols(q, start, end); // [seq, headDim]
      const kh = sliceCols(k, start, end);
      const vh = sliceCols(v, start, end);

      // Skaler çarpım ve transpoze işlemleri
      let scores = qh.matmul(kh.transpose()).mulScalar(scale); // [seq, seq]
      
      // Maskeleme operasyonu
      scores = applyCausalMask(scores);
      
      // GÜÇLENDİRME: NaN ve Gradyan Patlaması Koruması
      // Eğer scores matrisinin içinde tamamen maskelenmiş satırlar varsa softmax'ın çökmesini önleriz
      if (scores && scores.data) {
        for (let i = 0; i < scores.data.length; i++) {
          if (Number.isNaN(scores.data[i])) {
            scores.data[i] = -1e9; // NaN oluştuysa çok küçük bir değerle maskele
          }
        }
      }

      const weights = softmaxRows(scores); // [seq, seq]
      
      // GÜÇLENDİRME: Softmax sonrası NaN veya çökme kontrolü
      if (weights && weights.data) {
        for (let i = 0; i < weights.data.length; i++) {
          if (Number.isNaN(weights.data[i])) {
            weights.data[i] = 0; // Eğer tüm satır maskeliyse ve NaN çıktıysa 0'a eşitle
          }
        }
      }

      const headOut = weights.matmul(vh); // [seq, headDim]
      headOutputs.push(headOut);
    }

    const merged = concatCols(headOutputs); // [seq, dModel]
    return this.Wo.forward(merged);
  }
}