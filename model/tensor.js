// tensor.js — minimal 2D Tensor with autograd (reverse-mode, tape-free via closures)
import { rawMatmul, rawTranspose } from './matmul.js';
import { topoSort } from './autograd.js';

let TENSOR_ID = 0;

export class Tensor {
  constructor(data, shape, children = [], requiresGrad = false, label = '') {
    this.data = data instanceof Float64Array ? data : new Float64Array(data);
    this.shape = shape; // [rows, cols]
    this.children = children || [];
    
    // GÜNCELLEME: Eğer çocuklardan biri bile gradyan istiyorsa bu tensor de istemelidir.
    this.requiresGrad = requiresGrad || this.children.some(c => c && c.requiresGrad);
    
    // GÜNCELLEME: requiresGrad true ise HER ZAMAN gradyan dizisi oluşturulmalıdır.
    this.grad = this.requiresGrad ? new Float64Array(this.data.length) : null;
    this._backward = () => {};
    this.id = TENSOR_ID++;
    this.label = label;
  }

  static zeros(rows, cols, requiresGrad = false) {
    return new Tensor(new Float64Array(rows * cols), [rows, cols], [], requiresGrad);
  }

  get rows() { return this.shape[0]; }
  get cols() { return this.shape[1]; }

  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }

  // ---- elementwise ----
  add(other) {
    // GÜNCELLEME: Çıktı tensorünün gradyan isteyip istemediğini girdilerden anlıyoruz
    const requiresGrad = this.requiresGrad || other.requiresGrad;
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this, other], requiresGrad);
    
    for (let i = 0; i < this.data.length; i++) {
      out.data[i] = this.data[i] + other.data[i % other.data.length];
    }
    
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.data.length; i++) {
          let g = out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) this.grad[i] += g;
        }
      }
      if (other.requiresGrad && other.grad) {
        const len = other.data.length;
        for (let i = 0; i < this.data.length; i++) {
          let g = out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) other.grad[i % len] += g;
        }
      }
    };
    return out;
  }

  sub(other) {
    return this.add(other.mulScalar(-1));
  }

  mulElementwise(other) {
    const requiresGrad = this.requiresGrad || other.requiresGrad;
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this, other], requiresGrad);
    
    for (let i = 0; i < this.data.length; i++) out.data[i] = this.data[i] * other.data[i];
    
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.data.length; i++) {
          let g = other.data[i] * out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) this.grad[i] += g;
        }
      }
      if (other.requiresGrad && other.grad) {
        for (let i = 0; i < this.data.length; i++) {
          let g = this.data[i] * out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) other.grad[i] += g;
        }
      }
    };
    return out;
  }

  mulScalar(s) {
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this], this.requiresGrad);
    for (let i = 0; i < this.data.length; i++) out.data[i] = this.data[i] * s;
    
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.data.length; i++) {
          let g = s * out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) this.grad[i] += g;
        }
      }
    };
    return out;
  }

  // ---- matmul ----
  matmul(other) {
    const outData = rawMatmul(this.data, this.rows, this.cols, other.data, other.rows, other.cols);
    const requiresGrad = this.requiresGrad || other.requiresGrad;
    const out = new Tensor(outData, [this.rows, other.cols], [this, other], requiresGrad);
    
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        const bT = rawTranspose(other.data, other.rows, other.cols);
        const dA = rawMatmul(out.grad, out.rows, out.cols, bT, other.cols, other.rows);
        for (let i = 0; i < this.data.length; i++) {
          if (!Number.isNaN(dA[i]) && Number.isFinite(dA[i])) this.grad[i] += dA[i];
        }
      }
      if (other.requiresGrad && other.grad) {
        const aT = rawTranspose(this.data, this.rows, this.cols);
        const dB = rawMatmul(aT, this.cols, this.rows, out.grad, out.rows, out.cols);
        for (let i = 0; i < other.data.length; i++) {
          if (!Number.isNaN(dB[i]) && Number.isFinite(dB[i])) other.grad[i] += dB[i];
        }
      }
    };
    return out;
  }

  transpose() {
    const outData = rawTranspose(this.data, this.rows, this.cols);
    const out = new Tensor(outData, [this.cols, this.rows], [this], this.requiresGrad);
    
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        const gT = rawTranspose(out.grad, out.cols, out.rows);
        for (let i = 0; i < this.data.length; i++) {
          if (!Number.isNaN(gT[i]) && Number.isFinite(gT[i])) this.grad[i] += gT[i];
        }
      }
    };
    return out;
  }

  // ---- nonlinearities ----
  relu() {
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this], this.requiresGrad);
    for (let i = 0; i < this.data.length; i++) out.data[i] = Math.max(0, this.data[i]);
    
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.data.length; i++) {
          let g = (this.data[i] > 0 ? 1 : 0) * out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) this.grad[i] += g;
        }
      }
    };
    return out;
  }

  gelu() {
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this], this.requiresGrad);
    const c = Math.sqrt(2 / Math.PI);
    const cache = new Float64Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      const x = this.data[i];
      const t = Math.tanh(c * (x + 0.044715 * x * x * x));
      cache[i] = t;
      out.data[i] = 0.5 * x * (1 + t);
    }
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.data.length; i++) {
          const x = this.data[i];
          const t = cache[i];
          const dtdx = c * (1 + 3 * 0.044715 * x * x) * (1 - t * t);
          let deriv = 0.5 * (1 + t) + 0.5 * x * dtdx;
          if (Number.isNaN(deriv) || !Number.isFinite(deriv)) deriv = 0;
          let g = deriv * out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) this.grad[i] += g;
        }
      }
    };
    return out;
  }

  exp() {
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this], this.requiresGrad);
    for (let i = 0; i < this.data.length; i++) {
      // GÜÇLENDİRME: Gereksiz Math.getExponent kontrolü kaldırıldı.
      out.data[i] = Math.exp(Math.min(this.data[i], 88));
    }
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.data.length; i++) {
          let g = out.data[i] * out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) this.grad[i] += g;
        }
      }
    };
    return out;
  }

  log() {
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this], this.requiresGrad);
    for (let i = 0; i < this.data.length; i++) out.data[i] = Math.log(Math.max(this.data[i], 1e-15));
    
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.data.length; i++) {
          let g = (1 / Math.max(this.data[i], 1e-15)) * out.grad[i];
          if (!Number.isNaN(g) && Number.isFinite(g)) this.grad[i] += g;
        }
      }
    };
    return out;
  }

  // ---- reductions along rows ----
  sumRows() {
    const out = new Tensor(new Float64Array(this.rows), [this.rows, 1], [this], this.requiresGrad);
    for (let i = 0; i < this.rows; i++) {
      let s = 0;
      const offset = i * this.cols;
      for (let j = 0; j < this.cols; j++) {
        const val = this.data[offset + j];
        if (!Number.isNaN(val) && Number.isFinite(val)) s += val;
      }
      out.data[i] = s;
    }
    out._backward = () => {
      if (this.requiresGrad && this.grad) {
        for (let i = 0; i < this.rows; i++) {
          const g = out.grad[i];
          if (Number.isNaN(g) || !Number.isFinite(g)) continue;
          const offset = i * this.cols;
          for (let j = 0; j < this.cols; j++) this.grad[offset + j] += g;
        }
      }
    };
    return out;
  }

  divBroadcastCol(colVec) {
    const requiresGrad = this.requiresGrad || colVec.requiresGrad;
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this, colVec], requiresGrad);
    
    for (let i = 0; i < this.rows; i++) {
      let denom = colVec.data[i];
      if (Math.abs(denom) < 1e-15) denom = denom >= 0 ? 1e-15 : -1e-15;
      const offset = i * this.cols;
      for (let j = 0; j < this.cols; j++) out.data[offset + j] = this.data[offset + j] / denom;
    }
    out._backward = () => {
      for (let i = 0; i < this.rows; i++) {
        let denom = colVec.data[i];
        if (Math.abs(denom) < 1e-15) denom = denom >= 0 ? 1e-15 : -1e-15;
        let gradSumForColVec = 0;
        const offset = i * this.cols;
        for (let j = 0; j < this.cols; j++) {
          const idx = offset + j;
          let outG = out.grad[idx];
          if (Number.isNaN(outG) || !Number.isFinite(outG)) outG = 0;
          
          if (this.requiresGrad && this.grad) this.grad[idx] += outG / denom;
          gradSumForColVec += outG * (-this.data[idx] / (denom * denom));
        }
        if (colVec.requiresGrad && colVec.grad && !Number.isNaN(gradSumForColVec) && Number.isFinite(gradSumForColVec)) {
          colVec.grad[i] += gradSumForColVec;
        }
      }
    };
    return out;
  }

  subBroadcastCol(colVec) {
    const requiresGrad = this.requiresGrad || colVec.requiresGrad;
    const out = new Tensor(new Float64Array(this.data.length), this.shape, [this, colVec], requiresGrad);
    
    for (let i = 0; i < this.rows; i++) {
      const v = colVec.data[i];
      const offset = i * this.cols;
      for (let j = 0; j < this.cols; j++) out.data[offset + j] = this.data[offset + j] - v;
    }
    out._backward = () => {
      for (let i = 0; i < this.rows; i++) {
        let gSum = 0;
        const offset = i * this.cols;
        for (let j = 0; j < this.cols; j++) {
          const idx = offset + j;
          let outG = out.grad[idx];
          if (Number.isNaN(outG) || !Number.isFinite(outG)) outG = 0;
          if (this.requiresGrad && this.grad) this.grad[idx] += outG;
          gSum += outG;
        }
        if (colVec.requiresGrad && colVec.grad && !Number.isNaN(gSum) && Number.isFinite(gSum)) {
          colVec.grad[i] += -gSum;
        }
      }
    };
    return out;
  }

  backward() {
    const topo = topoSort(this);
    this.grad = this.grad || new Float64Array(this.data.length);
    this.grad.fill(1);
    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (t && t.requiresGrad) t._backward();
    }
  }
}