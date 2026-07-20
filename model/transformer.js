// transformer.js — GoobGPT: a tiny GPT-style decoder-only transformer
import { Module } from './module.js';
import { Embedding } from './embedding.js';
import { buildPositionalEncoding, addPositionalEncoding } from './positionalEncoding.js';
import { TransformerBlock } from './transformerBlock.js';
import { LayerNorm } from './layerNorm.js';
import { Linear } from './linear.js';

export class GoobGPT extends Module {
  constructor(cfg) {
    super();
    this.cfg = cfg;
    this.tokenEmbedding = new Embedding(cfg.vocabSize, cfg.dModel);
    this.peTable = buildPositionalEncoding(cfg.blockSize, cfg.dModel);
    this.blocks = [];
    for (let i = 0; i < cfg.numLayers; i++) {
      this.blocks.push(new TransformerBlock(cfg.dModel, cfg.numHeads));
    }
    this.lnFinal = new LayerNorm(cfg.dModel);
    this.head = new Linear(cfg.dModel, cfg.vocabSize, false);
  }

  forward(ids) {
    if (!ids || ids.length === 0) return null;
    let x = this.tokenEmbedding.forward(ids);
    x = addPositionalEncoding(x, this.peTable, this.cfg.dModel);
    for (const block of this.blocks) {
      if (block) x = block.forward(x);
    }
    x = this.lnFinal.forward(x);
    return this.head.forward(x);
  }

  stateDict() {
    return this.parameters().map(p => p && p.data ? Array.from(p.data) : []);
  }

  loadStateDict(arrays) {
    const params = this.parameters();
    if (params.length !== arrays.length) {
      throw new Error(`Checkpoint has ${arrays.length} tensors, model has ${params.length}`);
    }
    params.forEach((p, i) => {
      if (p && p.data && arrays[i]) {
        p.data.set(Float64Array.from(arrays[i]));
      }
    });
  }
}