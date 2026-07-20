// backward.js — run a full backward pass from a scalar loss Tensor
export function backward(lossTensor) {
  // GÜÇLENDİRME 1: Geçersiz veya null Tensor koruması
  if (!lossTensor) {
    throw new Error('[GOOBER ENGINE ERROR] backward() can only be called on a valid Tensor instance.');
  }

  // GÜÇLENDİRME 2: Gradyan Başlatma (Seed Gradient) Kontrolü
  // Kayıp (loss) skalar bir değerdir. Geriye yayılımın doğru tetiklenmesi için 
  // lossTensor'ın kendi gradyanının (grad) başlangıçta tam olarak 1.0 olması gerekir.
  if (lossTensor.grad === undefined || lossTensor.grad === null || lossTensor.grad === 0) {
    lossTensor.grad = 1.0; 
  } else if (typeof lossTensor.grad === 'object' && lossTensor.grad.data) {
    // Eğer grad bir matris/tensor nesnesiyse içindeki veriyi 1.0 ile doldur
    for (let i = 0; i < lossTensor.grad.data.length; i++) {
      lossTensor.grad.data[i] = 1.0;
    }
  }

  // GÜÇLENDİRME 3: Sayısal Kararlılık (NaN / Infinity) Kontrolü
  // Eğer loss tensorunun kendisi patladıysa (NaN olduysa), geriye yayılımı yapıp 
  // tüm modelin ağırlıklarını (weights) bozmasını engellemek için koruma sağlıyoruz.
  if (lossTensor.data) {
    let hasNan = false;
    if (Array.isArray(lossTensor.data)) {
      hasNan = lossTensor.data.some(v => Number.isNaN(v) || !Number.isFinite(v));
    } else if (typeof lossTensor.data === 'number') {
      hasNan = Number.isNaN(lossTensor.data) || !Number.isFinite(lossTensor.data);
    }

    if (hasNan) {
      console.warn('[GOOBER ENGINE WARNING] Loss is NaN or Infinite. Skipping backward pass to protect weights.');
      return; // Ağırlıkların patlamasını önlemek için geri yayılımı pas geç
    }
  }

  // GÜÇLENDİRME 4: Hata Yakalama (Graceful Fallback)
  try {
    if (typeof lossTensor.backward === 'function') {
      lossTensor.backward();
    } else {
      throw new Error('lossTensor does not implement backward() method.');
    }
  } catch (error) {
    console.error('[GOOBER ENGINE CRITICAL ERROR] Backward pass failed:', error);
  }
}