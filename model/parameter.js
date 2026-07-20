// parameter.js — a Tensor that is always a trainable leaf node
import { Tensor } from './tensor.js';

export class Parameter extends Tensor {
  constructor(data, shape) {
    super(data, shape, [], true);
    this.isParameter = true;
  }
}