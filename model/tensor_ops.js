// tensor_ops.js — composite ops built on top of tensor.js primitives
import { Tensor } from './tensor.js';

// Adds -Infinity (approx) above the diagonal so softmax zeroes out future positions.
export function applyCausalMask(scores) {
  if (!scores || !scores.rows) return scores;
  const n = scores.rows; // scores is [seqLen, seqLen]
  const maskData = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      maskData[i * n + j] = j > i ? -1e9 : 0;
    }
  }
  const mask = new Tensor(maskData, [n, n], [], false);
  return scores.add(mask);
}

// Slice a contiguous range of columns out of a [rows, cols] tensor -> [rows, end-start]
export function sliceCols(t, start, end) {
  if (!t || !t.data) return t;
  const width = end - start;
  const out = new Tensor(new Float64Array(t.rows * width), [t.rows, width], [t]);
  for (let i = 0; i < t.rows; i++) {
    const tOffset = i * t.cols;
    const outOffset = i * width;
    for (let j = 0; j < width; j++) {
      out.data[outOffset + j] = t.data[tOffset + start + j];
    }
  }
  out._backward = () => {
    if (t.requiresGrad) {
      if (!t.grad) t.grad = new Float64Array(t.data.length);
      for (let i = 0; i < t.rows; i++) {
        const tOffset = i * t.cols;
        const outOffset = i * width;
        for (let j = 0; j < width; j++) {
          let g = out.grad[outOffset + j];
          if (!Number.isNaN(g) && Number.isFinite(g)) {
            t.grad[tOffset + start + j] += g;
          }
        }
      }
    }
  };
  return out;
}

// Concatenate several [rows, w_k] tensors column-wise -> [rows, sum(w_k)]
export function concatCols(tensors) {
  if (!tensors || tensors.length === 0) return null;
  const validTensors = tensors.filter(t => t && t.data);
  if (validTensors.length === 0) return null;

  const rows = validTensors[0].rows;
  const totalCols = validTensors.reduce((s, t) => s + t.cols, 0);
  const out = new Tensor(new Float64Array(rows * totalCols), [rows, totalCols], validTensors);
  
  let colOffset = 0;
  const offsets = [];
  for (const t of validTensors) {
    offsets.push(colOffset);
    for (let i = 0; i < rows; i++) {
      const outOffset = i * totalCols + colOffset;
      const tOffset = i * t.cols;
      for (let j = 0; j < t.cols; j++) {
        out.data[outOffset + j] = t.data[tOffset + j];
      }
    }
    colOffset += t.cols;
  }
  out._backward = () => {
    validTensors.forEach((t, idx) => {
      if (!t.requiresGrad) return;
      if (!t.grad) t.grad = new Float64Array(t.data.length);
      const off = offsets[idx];
      for (let i = 0; i < rows; i++) {
        const outOffset = i * totalCols + off;
        const tOffset = i * t.cols;
        for (let j = 0; j < t.cols; j++) {
          let g = out.grad[outOffset + j];
          if (!Number.isNaN(g) && Number.isFinite(g)) {
            t.grad[tOffset + j] += g;
          }
        }
      }
    });
  };
  return out;
}