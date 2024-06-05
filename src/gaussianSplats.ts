import { BYTES_F32, BYTES_U8 } from './constants.ts';

// TODO .PLY loader: https://github.com/antimatter15/splat/blob/main/main.js#L474

export const SPLAT_SIZE_BYTES =
  3 * BYTES_F32 + // position
  3 * BYTES_F32 + // scale
  4 * BYTES_U8 + // color (SH C0)
  4 * BYTES_U8; // rotation

export const VERTICES_PER_SPLAT = 6;

export interface GaussianSplats {
  count: number;

  // data for Gaussians:
  positions: Float32Array; // vec4 * count
  positionsBuffer: GPUBuffer;
  rotationsBuffer: GPUBuffer;
  scalesBuffer: GPUBuffer;
  colorsBuffer: GPUBuffer; // TODO this can be inside vec4<u32> with quaternion

  // indices:
  /** indices used for rendering. See `VERTICES_PER_SPLAT` */
  indicesBuffer: GPUBuffer;
}
