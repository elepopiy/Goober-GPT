// transformerBlock.js — pre-norm transformer decoder block with residual connections
import { Module } from './module.js';
import { MultiHeadAttention } from './attention.js';
import { FeedForward } from './feedForward.js';
import { LayerNorm } from './layerNorm.js';

export class TransformerBlock extends Module {
  constructor(dModel, numHeads) {
    super();
    this.ln1 = new LayerNorm(dModel);
    this.attn = new MultiHeadAttention(dModel, numHeads);
    this.ln2 = new LayerNorm(dModel);
    this.ff = new FeedForward(dModel);
  }

  forward(x) {
    if (!x) return x;
    const attnOut = this.attn.forward(this.ln1.forward(x));
    const x2 = x.add(attnOut);
    const ffOut = this.ff.forward(this.ln2.forward(x2));
    return x2.add(ffOut);
  }
}