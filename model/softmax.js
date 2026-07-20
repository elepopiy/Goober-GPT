// softmax.js — numerically stable row-wise softmax, built from Tensor primitives
import { Tensor } from './tensor.js';

function detachedRowMax(t) {
  const out = new Float64Array(t.rows);
  for (let i = 0; i < t.rows; i++) {
    let m = -Infinity;
    const offset = i * t.cols;
    for (let j = 0; j < t.cols; j++) {
      const val = t.data[offset + j];
      if (!Number.isNaN(val) && Number.isFinite(val)) {
        m = Math.max(m, val);
      }
    }
    // GÜÇLENDİRME: Eğer tüm satır bozuksa çökmeyi önlemek için 0'a çek
    if (m === -Infinity) m = 0;
    out[i] = m;
  }
  return new Tensor(out, [t.rows, 1], [], false);
}

export function softmaxRows(t) {
  if (!t || !t.data || t.data.length === 0) return t;
  const maxCol = detachedRowMax(t);
  const shifted = t.subBroadcastCol(maxCol);
  const exps = shifted.exp();
  const sums = exps.sumRows();
  return exps.divBroadcastCol(sums);
}