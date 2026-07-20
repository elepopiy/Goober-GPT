// config.js — hyperparameters + Goob persona settings

export const modelConfig = {
  dModel: 64,         // 256'dan 64'e düşürdük (İşlem yükünü 16 kat azaltır!)
  numHeads: 8,         // 8'den 4'e düşürdük
  numLayers: 6,        // 6'dan 3'e düşürdük (Katman ağırlığını yarıya indirdik)
  blockSize: 128,       // 64'ten 32'ye düşürdük (Attention hesaplamalarını 4 kat hızlandırır)
  learningRate: 0.001, // Hızlı overfit için ideal oran (1e-3)
};

export const trainingConfig = {
  steps: 20000,         // Küçük veri seti için 5000 step roket hızında biter
  logEvery: 1,        // Konsol log şişmesini engellemek için 50 yaptık
  saveEvery: 5,      // Disk yazma gecikmesini azaltmak için 500 yaptık
};

export const paths = {
  trainText: './data/train.txt',
  vocab: './data/vocab.json',
  checkpoint: './saves/goob-gpt.json',
};

// Goob's personality — used to flavor generation prompts in chat.js.
// Goob loves hugs, is soft, round, cheerful, and endlessly affectionate.
export const persona = {
  name: 'Goob',
  greeting: "*wobbles over happily* hii!! it's me, Goob! wanna hug?? :3",
  farewell: '*gives one last big squishy hug* byebye, come back soon!! <3',
};