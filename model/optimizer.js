// optimizer.js — Adam optimizer over a flat list of Parameters
export class Adam {
  constructor(parameters, lr = 0.003, betas = [0.9, 0.999], eps = 1e-8) {
    this.parameters = parameters.filter(p => p !== null && p !== undefined);
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.t = 0;
    this.m = this.parameters.map(p => new Float64Array(p.data.length));
    this.v = this.parameters.map(p => new Float64Array(p.data.length));
  }

  step() {
    this.t += 1;
    const { beta1, beta2, eps, lr, t } = this;
    const biasCorr1 = 1 - Math.pow(beta1, t);
    const biasCorr2 = 1 - Math.pow(beta2, t);
    
    // GÜÇLENDİRME: Global Gradient Clipping Eşiği
    const clipValue = 5.0;

    this.parameters.forEach((p, idx) => {
      // Eğer gradyan dizisi henüz oluşturulmadıysa temiz bir dizi ata
      if (!p.grad) {
        p.grad = new Float64Array(p.data.length);
      }

      const m = this.m[idx];
      const v = this.v[idx];
      
      for (let i = 0; i < p.data.length; i++) {
        let g = p.grad[i];
        
        // GÜÇLENDİRME: Gradyan Temizleme ve Kırpma (Clipping)
        // Gradyanın NaN olmasını engelle, aşırı büyük değerleri limitlerde tut
        if (Number.isNaN(g) || !Number.isFinite(g)) g = 0.0;
        if (g > clipValue) g = clipValue;
        if (g < -clipValue) g = -clipValue;

        // Adam güncelleme denklemleri
        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;
        
        const mHat = m[i] / biasCorr1;
        const vHat = v[i] / biasCorr2;
        
        const delta = (lr * mHat) / (Math.sqrt(Math.max(vHat, 0)) + eps);
        
        // Ağırlıkların bozulmasını önlemek için değişim miktarını da denetle
        if (!Number.isNaN(delta) && Number.isFinite(delta)) {
          p.data[i] -= delta;
        }
      }
    });
  }

  zeroGrad() {
    for (const p of this.parameters) {
      if (p && typeof p.zeroGrad === 'function') p.zeroGrad();
    }
  }
}