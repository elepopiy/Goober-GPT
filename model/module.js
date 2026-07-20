// module.js — lightweight base class (loosely mirrors nn.Module)
export class Module {
  parameters() {
    let params = [];
    for (const key of Object.keys(this)) {
      try {
        const val = this[key];
        if (!val) continue;
        
        if (val.isParameter) {
          params.push(val);
        } else if (val instanceof Module) {
          params = params.concat(val.parameters());
        } else if (Array.isArray(val)) {
          for (const item of val) {
            if (item instanceof Module) params = params.concat(item.parameters());
          }
        }
      } catch (e) {
        console.warn(`[GOOBER ENGINE] Warning checking parameters for key ${key}:`, e);
      }
    }
    return params;
  }

  zeroGrad() {
    for (const p of this.parameters()) {
      if (p && typeof p.zeroGrad === 'function') p.zeroGrad();
    }
  }
}