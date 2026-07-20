// train.js — builds/loads vocab, builds the model, and trains GoobGPT on data/train.txt
import fs from 'fs';
import { GoobGPT } from './model/transformer.js';
import { Tokenizer } from './model/tokenizer.js';
import { Trainer } from './model/trainer.js';
import { modelConfig, trainingConfig, paths } from './config.js';

function main() {
  if (!fs.existsSync(paths.trainText)) {
    console.error(`Missing training file: ${paths.trainText}`);
    process.exit(1);
  }
  const text = fs.readFileSync(paths.trainText, 'utf-8');

  // Build (or reuse) the vocabulary
  let tokenizer;
  if (fs.existsSync(paths.vocab)) {
    const vocabJson = JSON.parse(fs.readFileSync(paths.vocab, 'utf-8'));
    tokenizer = Tokenizer.fromJSON(vocabJson);
    console.log(`loaded vocab (${tokenizer.vocabSize} chars) from ${paths.vocab}`);
  } else {
    tokenizer = Tokenizer.buildFromText(text);
    fs.writeFileSync(paths.vocab, JSON.stringify(tokenizer.toJSON(), null, 2));
    console.log(`built vocab (${tokenizer.vocabSize} chars), saved to ${paths.vocab}`);
  }

  const cfg = { ...modelConfig, vocabSize: tokenizer.vocabSize };
  const tokenIds = tokenizer.encode(text);
  console.log(`corpus: ${tokenIds.length} tokens, vocabSize=${cfg.vocabSize}, blockSize=${cfg.blockSize}`);

  let model;
  if (fs.existsSync(paths.checkpoint)) {
    console.log(`found existing checkpoint at ${paths.checkpoint}, resuming training...`);
    const ckpt = JSON.parse(fs.readFileSync(paths.checkpoint, 'utf-8'));
    model = new GoobGPT(ckpt.cfg);
    model.loadStateDict(ckpt.state);
  } else {
    model = new GoobGPT(cfg);
    console.log('initialized a fresh GoobGPT model');
  }

  const trainer = new Trainer(model, tokenIds, cfg, paths.checkpoint);
  trainer.train(trainingConfig.steps, {
    logEvery: trainingConfig.logEvery,
    saveEvery: trainingConfig.saveEvery,
  });

  console.log('training complete! run `node chat.js` (or chat.bat) to talk with Goob.');
}

main();
