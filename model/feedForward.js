// feedForward.js — position-wise MLP with GELU activation
import { Module } from './module.js';
import { Linear } from './linear.js';

export class FeedForward extends Module {
  constructor(dModel, hiddenMultiplier = 4) {
    super();
    const hidden = dModel * hiddenMultiplier;
    this.fc1 = new Linear(dModel, hidden);
    this.fc2 = new Linear(hidden, dModel);
  }

  forward(x) {
    // GÜÇLENDİRME 1: Giriş tensorünün geçerlilik kontrolü
    if (!x || !x.data) return x;

    // İlk linear katman ileri geçişi
    const h1 = this.fc1.forward(x);
    
    // GÜÇLENDİRME 2: GELU öncesi patlama kontrolü
    // Eğer h1 içinde aşırı büyük veya bozuk değerler varsa temizle/sınırla
    if (h1 && h1.data) {
      for (let i = 0; i < h1.data.length; i++) {
        if (Number.isNaN(h1.data[i])) {
          h1.data[i] = 0.0;
        } else if (h1.data[i] > 1e4) {
          h1.data[i] = 1e4; // Infinity'e gitmesini engellemek için tavan kırpması
        } else if (h1.data[i] < -1e4) {
          h1.data[i] = -1e4;
        }
      }
    }

    // Orijinal akış korundu: .gelu() aktivasyonu uygulanıyor
    const h = h1.gelu(); 

    // GÜÇLENDİRME 3: GELU sonrası NaN/Infinity temizliği
    if (h && h.data) {
      for (let i = 0; i < h.data.length; i++) {
        if (Number.isNaN(h.data[i]) || !Number.isFinite(h.data[i])) {
          h.data[i] = 0.0; // Aktivasyon çöktüyse nöronu güvenli bir şekilde sustur
        }
      }
    }

    // İkinci katmana güvenli veriyi aktar
    return this.fc2.forward(h);
  }
}