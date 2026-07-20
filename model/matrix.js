// matrix.js — plain flat-array matrix helpers (row-major, shape = [rows, cols])

export function zeros(rows, cols) {
  return new Float64Array(rows * cols);
}

export function randMatrix(rows, cols, scale = 0.08) {
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) {
    data[i] = (Math.random() * 2 - 1) * scale;
    if (Number.isNaN(data[i])) data[i] = 0; // Aşırı önlem
  }
  return data;
}

export function fill(rows, cols, value) {
  const data = new Float64Array(rows * cols);
  data.fill(value);
  return data;
}

export function clone(data) {
  if (!data) return new Float64Array(0);
  return new Float64Array(data);
}