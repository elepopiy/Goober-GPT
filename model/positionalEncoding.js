// positionalEncoding.js — fixed sinusoidal positional encoding, added to token embeddings
import { Tensor } from './tensor.js';

export function buildPositionalEncoding(maxLen, dim) {
  // GÜÇLENDİRME: Geçersiz boyut koruması
  if (!maxLen || !dim || maxLen <= 0 || dim <= 0) {
    return new Float64Array(0);
  }
  
  const data = new Float64Array(maxLen * dim);
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dim; i++) {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
      let val = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      
      // GÜÇLENDİRME: Sayısal kararlılık kontrolü
      if (Number.isNaN(val) || !Number.isFinite(val)) val = 0;
      data[pos * dim + i] = val;
    }
  }
  return data;
}

// x: Tensor [seqLen, dim]; peTable: Float64Array from buildPositionalEncoding
export function addPositionalEncoding(x, peTable, dim) {
  if (!x || !x.data || !peTable || peTable.length === 0) return x;
  
  const seqLen = x.rows;
  const totalElements = seqLen * dim;
  
  // GÜÇLENDİRME: Boyut sınır taşması kontrolü
  if (totalElements > peTable.length) {
    return x; 
  }
  
  const peSlice = new Tensor(peTable.subarray(0, totalElements), [seqLen, dim], [], false);
  return x.add(peSlice);
}