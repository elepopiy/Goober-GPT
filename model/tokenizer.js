// tokenizer.js — improved word tokenizer with special token support

export class Tokenizer {
  constructor(vocab = []) {
    this.vocab = [...vocab];

    // Özel tokenlar
    const specialTokens = [
      "<PAD>",
      "<UNK>",
      "<USER>",
      "<GOOB>",
      "<END>"
    ];

    for (const token of specialTokens) {
      if (!this.vocab.includes(token)) {
        this.vocab.unshift(token);
      }
    }

    this.wordToId = new Map();

    this.vocab.forEach((word, i) => {
      this.wordToId.set(word, i);
    });

    this.unkId = this.wordToId.get("<UNK>");
  }

  static tokenize(text) {
    if (typeof text !== "string") return [];

    // Özel tokenları ve kelimeleri ayır
    return (
      text.match(
        /<PAD>|<UNK>|<USER>|<GOOB>|<END>|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\s]/g
      ) || []
    );
  }

  static buildFromText(text) {
    const words = Tokenizer.tokenize(text);

    const vocab = [...new Set(words)];

    return new Tokenizer(vocab);
  }

  static fromJSON(json) {
    return new Tokenizer(json?.vocab ?? []);
  }

  toJSON() {
    return {
      vocab: this.vocab
    };
  }

  get vocabSize() {
    return this.vocab.length;
  }

  encode(text) {
    const words = Tokenizer.tokenize(text);

    return words.map(word => {
      if (this.wordToId.has(word)) {
        return this.wordToId.get(word);
      }

      return this.unkId;
    });
  }

  decode(ids) {
    if (!Array.isArray(ids)) return "";

    let out = "";

    for (const id of ids) {
      const token = this.vocab[id] ?? "<UNK>";

      // Noktalama öncesi boşluk bırakma
      if (/^[.,!?;:]$/.test(token)) {
        out += token;
      } else {
        if (out.length) out += " ";
        out += token;
      }
    }

    return out;
  }
}