// layerNorm.js — row-wise layer normalization with learnable gain/bias
import { Module } from './module.js';
import { Parameter } from './parameter.js';
import { Tensor } from './tensor.js';
import { fill, zeros } from './matrix.js';

function sqrtTensor(t, eps) {
  const out = new Tensor(new Float64Array(t.data.length), t.shape, [t]);
  for (let i = 0; i < t.data.length; i++) {
    out.data[i] = Math.sqrt(Math.max(t.data[i] + eps, 0)); // Negatif karekök koruması
  }
  out._backward = () => {
    if (t.requiresGrad) {
      if (!t.grad) t.grad = new Float64Array(t.data.length);
      for (let i = 0; i < t.data.length; i++) {
        // GÜÇLENDİRME: Sıfıra bölme ve NaN gradyan sızıntısı koruması
        const denom = out.data[i] < 0 ? out.data[i] - 1e-12 : out.data[i] + 1e-12;
        const g = (0.5 / denom) * out.grad[i];
        
        if (!Number.isNaN(g) && Number.isFinite(g)) {
          t.grad[i] += g;
        }
      }
    }
  };
  return out;
}

export class LayerNorm extends Module {
  constructor(dim, eps = 1e-5) {
    super();
    this.dim = dim;
    this.eps = eps;
    this.gamma = new Parameter(fill(1, dim, 1), [1, dim]);
    this.beta = new Parameter(zeros(1, dim), [1, dim]);
  }

  forward(x) {
    if (!x || !x.data) return x;
    const n = x.cols;
    
    const mean = x.sumRows().mulScalar(1 / n); // [rows,1]
    const centered = x.subBroadcastCol(mean); // [rows,dim]
    const sq = centered.mulElementwise(centered);
    const variance = sq.sumRows().mulScalar(1 / n); // [rows,1]
    const std = sqrtTensor(variance, this.eps); // [rows,1]
    const normalized = centered.divBroadcastCol(std); // [rows,dim]
    
    // Orijinal zincir korundu
    return normalized.mulElementwise(this._broadcastRow(this.gamma, x.rows)).add(this.beta);
  }

  _broadcastRow(param, rows) {
    const out = new Tensor(new Float64Array(rows * this.dim), [rows, this.dim], [param]);
    for (let i = 0; i < rows; i++) out.data.set(param.data, i * this.dim);
    out._backward = () => {
      if (param.requiresGrad) {
        if (!param.grad) param.grad = new Float64Array(param.data.length);
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < this.dim; j++) {
            const g = out.grad[i * this.dim + j];
            if (!Number.isNaN(g) && Number.isFinite(g)) {
              param.grad[j] += g;
            }
          }
        }
      }
    };
    return out;
  }
}