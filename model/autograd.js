// autograd.js — topological ordering utility shared by Tensor.backward()
// (kept as its own module so the graph-walking logic is easy to find/reuse)

export function topoSort(root) {
  const topo = [];
  const visited = new Set();
  
  // GÜÇLENDİRME: Stack Overflow'u (Maksimum Call Stack Aşımını) önlemek için iteratif yaklaşım
  // JavaScript'in çağrı yığıtı (call stack) yerine kendi dizimizi (stack) yönetiyoruz.
  const stack = [{ node: root, childIndex: 0 }];
  const processing = new Set(); // Döngüsel grafikleri (cycles) yakalamak için koruma

  while (stack.length > 0) {
    const current = stack[stack.length - 1];
    const t = current.node;

    // Geçersiz veya null düğüm koruması
    if (!t || t.id === undefined) {
      stack.pop();
      continue;
    }

    // Eğer bu düğüm daha önce tamamen işlendiyse stack'ten çıkar
    if (visited.has(t.id)) {
      stack.pop();
      continue;
    }

    const children = t.children || [];

    // Eğer çocukları henüz taranmadıysa ve ilk kez buraya geliyorsak döngü kontrolü yap
    if (current.childIndex === 0) {
      if (processing.has(t.id)) {
        // Döngüsel bir grafik algılandı (A -> B -> A gibi). Çökmeyi önlemek için atla.
        stack.pop();
        continue;
      }
      processing.add(t.id);
    }

    // Çocukları tek tek gez
    if (current.childIndex < children.length) {
      const child = children[current.childIndex];
      current.childIndex++; // Bir sonraki dönüşte sonraki çocuğa geçmek için index'i artır
      
      if (child && child.id !== undefined && !visited.has(child.id)) {
        stack.push({ node: child, childIndex: 0 });
      }
    } else {
      // Düğümün tüm çocukları başarıyla gezildi (Post-order sonu)
      processing.delete(t.id);
      visited.add(t.id);
      topo.push(t);
      stack.pop();
    }
  }

  return topo;
}