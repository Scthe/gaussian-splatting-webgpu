import { Mat4 } from 'wgpu-matrix';
import { GpuProfiler } from '../gpuProfiler.ts';
import { Dimensions } from '../utils.ts';
import { GaussianSplats } from '../gaussianSplats.ts';

export interface PassCtx {
  device: GPUDevice;
  cmdBuf: GPUCommandEncoder;
  mvpMatrix: Mat4;
  viewMatrix: Mat4;
  projMatrix: Mat4;
  profiler: GpuProfiler | undefined;
  viewport: Dimensions;
  focalX: number;
  focalY: number;
  splats: GaussianSplats;
}
