// linear.js — y = x @ W + b
import { Module } from './module.js';
import { Parameter } from './parameter.js';
import { randMatrix, zeros } from './matrix.js';

export class Linear extends Module {
  constructor(inFeatures, outFeatures, bias = true) {
    super();
    const scale = 1 / Math.sqrt(inFeatures);
    this.W = new Parameter(randMatrix(inFeatures, outFeatures, scale), [inFeatures, outFeatures]);
    this.useBias = bias;
    if (bias) this.b = new Parameter(zeros(1, outFeatures), [1, outFeatures]);
  }

  forward(x) {
    // GÜÇLENDİRME: Girdi kontrolü ve boyut uyuşmazlığı güvenliği
    if (!x || !x.data) return x;
    
    // Matris çarpımı gereksinimi: x.cols ile this.W.shape[0] eşit olmalı
    if (x.cols !== this.W.shape[0]) {
      console.warn(`[GOOBER ENGINE WARNING] Dimension mismatch in Linear layer. Expected ${this.W.shape[0]} columns, got ${x.cols}.`);
      return x; 
    }

    let out = x.matmul(this.W);
    if (this.useBias && this.b) out = out.add(this.b);
    return out;
  }
}