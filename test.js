// test.js — quick sanity checks: run with `node test.js`
import { Tensor } from './model/tensor.js';
import { GoobGPT } from './model/transformer.js';
import { crossEntropyLoss } from './model/loss.js';
import { Adam } from './model/optimizer.js';
import { Tokenizer } from './model/tokenizer.js';
import { generate } from './model/generator.js';

let passed = 0;
let failed = 0;
function check(name, cond) {
  if (cond) {
    console.log(`  ok  - ${name}`);
    passed++;
  } else {
    console.log(`FAIL  - ${name}`);
    failed++;
  }
}

console.log('1) matmul + backward shape check');
{
  const a = new Tensor([1, 2, 3, 4, 5, 6], [2, 3], [], true);
  const b = new Tensor([1, 0, 0, 1, 1, 1], [3, 2], [], true);
  const c = a.matmul(b);
  check('output shape is [2,2]', c.rows === 2 && c.cols === 2);
  c.backward();
  check('grad populated on a', a.grad.some(v => v !== 0));
  check('grad populated on b', b.grad.some(v => v !== 0));
}

console.log('2) numerical gradient check on a small op chain');
{
  function loss(xVal) {
    const x = new Tensor([xVal], [1, 1], [], true);
    const y = x.mulScalar(3).add(new Tensor([2], [1, 1])).relu();
    return { x, y };
  }
  const { x, y } = loss(2.0);
  y.backward();
  const analytic = x.grad[0];
  const eps = 1e-4;
  const y1 = loss(2.0 + eps).y.data[0];
  const y0 = loss(2.0 - eps).y.data[0];
  const numeric = (y1 - y0) / (2 * eps);
  check(`analytic grad (${analytic.toFixed(4)}) ~= numeric grad (${numeric.toFixed(4)})`, Math.abs(analytic - numeric) < 1e-2);
}

console.log('3) full model forward/backward + overfit check');
{
  const cfg = { vocabSize: 10, dModel: 16, numHeads: 2, numLayers: 2, blockSize: 6, learningRate: 0.02 };
  const model = new GoobGPT(cfg);
  const opt = new Adam(model.parameters(), cfg.learningRate);
  const input = [1, 2, 3, 4, 5, 6];
  const target = [2, 3, 4, 5, 6, 7];
  let firstLoss = null;
  let lastLoss = null;
  for (let i = 0; i < 25; i++) {
    opt.zeroGrad();
    const logits = model.forward(input);
    const loss = crossEntropyLoss(logits, target);
    if (i === 0) firstLoss = loss.data[0];
    loss.backward();
    opt.step();
    lastLoss = loss.data[0];
  }
  check(`loss decreased (first=${firstLoss.toFixed(3)} -> last=${lastLoss.toFixed(3)})`, lastLoss < firstLoss * 0.5);
}

console.log('4) tokenizer round-trip');
{
  const tok = Tokenizer.buildFromText('hug the goob!!');
  const ids = tok.encode('goob');
  const decoded = tok.decode(ids);
  check('encode/decode round-trip matches', decoded === 'goob');
}

console.log('5) generator produces text of requested length');
{
  const tok = Tokenizer.buildFromText('hii hug goob teehee');
  const cfg = { vocabSize: tok.vocabSize, dModel: 8, numHeads: 2, numLayers: 1, blockSize: 16, learningRate: 0.01 };
  const model = new GoobGPT(cfg);
  const out = generate(model, tok, 'hii', { maxNewTokens: 10, temperature: 1.0, topK: 5 });
  check('generated string length >= prompt length', out.length >= 3);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
