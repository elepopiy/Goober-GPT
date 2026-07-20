// embedding.js — token embedding table with gather (forward) / scatter-add (backward)
import { Module } from './module.js';
import { Parameter } from './parameter.js';
import { Tensor } from './tensor.js';
import { randMatrix } from './matrix.js';

export class Embedding extends Module {
  constructor(vocabSize, dim) {
    super();
    this.vocabSize = vocabSize;
    this.dim = dim;
    this.table = new Parameter(randMatrix(vocabSize, dim, 0.08), [vocabSize, dim]);
  }

  // ids: plain array of integers, length = seqLen
  forward(ids) {
    // GÜÇLENDİRME 1: Geçersiz giriş dizisi koruması
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      // Eğer boş veya geçersizse sıfır matrisli güvenli bir tensor dön
      return new Tensor(new Float64Array(1 * this.dim), [1, this.dim], [this.table]);
    }

    const dim = this.dim;
    const vocabSize = this.vocabSize;
    const out = new Tensor(new Float64Array(ids.length * dim), [ids.length, dim], [this.table]);
    
    for (let i = 0; i < ids.length; i++) {
      let id = ids[i];
      
      // GÜÇLENDİRME 2: Geçersiz Token/Sınır Aşımı Koruması (Out-of-Bounds Protection)
      // Eğer gelen id vocab boyutundan büyükse veya negatifse çökmemesi için 0. indexe (genelde padding/unknown) yönlendir.
      if (id < 0 || id >= vocabSize || typeof id !== 'number' || Number.isNaN(id)) {
        id = 0; 
      }

      const start = id * dim;
      const end = start + dim;
      out.data.set(this.table.data.subarray(start, end), i * dim);
    }
    
    out._backward = () => {
      if (this.table.requiresGrad) {
        // GÜÇLENDİRME 3: Grad dizisinin tanımlı olduğundan emin ol
        if (!this.table.grad) {
          this.table.grad = new Float64Array(this.table.data.length);
        }

        for (let i = 0; i < ids.length; i++) {
          let id = ids[i];
          if (id < 0 || id >= vocabSize || typeof id !== 'number' || Number.isNaN(id)) {
            id = 0;
          }

          const rowOff = id * dim;
          const outOff = i * dim;

          for (let j = 0; j < dim; j++) {
            const g = out.grad[outOff + j];
            
            // GÜÇLENDİRME 4: NaN Gradyan Filtresi
            // Gradyan patlaması sırasında üst katmanlardan NaN gelirse, tablodaki ağırlıkları koru.
            if (!Number.isNaN(g) && Number.isFinite(g)) {
              this.table.grad[rowOff + j] += g;
            }
          }
        }
      }
    };
    return out;
  }
}