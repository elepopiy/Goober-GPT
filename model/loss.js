// loss.js — mean cross-entropy loss over a sequence of predictions
import { Tensor } from './tensor.js';
import { softmaxRows } from './softmax.js';

// logits: Tensor [seqLen, vocabSize], targets: array of correct token ids, length seqLen
export function crossEntropyLoss(logits, targets, mask = null) {
  if (!logits || !logits.data || !targets || targets.length === 0) {
    return new Tensor(new Float64Array([0]), [1, 1], [], false);
  }

  const probs = softmaxRows(logits);
  if (probs && probs.data) {
    for (let i = 0; i < probs.data.length; i++) {
      probs.data[i] = Math.max(probs.data[i], 1e-15);
    }
  }

  const logProbs = probs.log();
  const seqLen = logits.rows;
  const vocab = logits.cols;
  const onehot = new Float64Array(seqLen * vocab);

  for (let i = 0; i < seqLen; i++) {
    if (mask && !mask[i]) continue; // maskelenmiş pozisyon -> onehot tamamen sıfır kalır
    let targetId = targets[i];
    if (targetId < 0 || targetId >= vocab || typeof targetId !== 'number' || Number.isNaN(targetId)) {
      targetId = 0;
    }
    onehot[i * vocab + targetId] = 1;
  }

  const onehotTensor = new Tensor(onehot, [seqLen, vocab], [], false);
  const picked = logProbs.mulElementwise(onehotTensor);
  const perRow = picked.sumRows();

  // ÖNEMLİ: artık seqLen'e değil, GERÇEKTEN sayılan pozisyon sayısına bölüyoruz
  let sumData = 0;
  let countedPositions = 0;
  for (let i = 0; i < seqLen; i++) {
    if (mask && !mask[i]) continue;
    const val = perRow.data[i];
    if (!Number.isNaN(val) && Number.isFinite(val)) {
      sumData += val;
      countedPositions++;
    }
  }
  const denom = countedPositions > 0 ? countedPositions : 1;

  const meanNeg = new Tensor(new Float64Array([-sumData / denom]), [1, 1], [perRow]);
  meanNeg._backward = () => {
    if (perRow.requiresGrad) {
      if (!perRow.grad) perRow.grad = new Float64Array(perRow.data.length);
      const g = meanNeg.grad[0] * (-1 / denom);
      if (!Number.isNaN(g) && Number.isFinite(g)) {
        for (let i = 0; i < seqLen; i++) {
          if (mask && !mask[i]) continue; // maskeli pozisyonlara gradyan sızdırma
          perRow.grad[i] += g;
        }
      }
    }
  };
  return meanNeg;
}