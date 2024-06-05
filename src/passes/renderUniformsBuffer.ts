import { BYTES_MAT4, BYTES_VEC4, CONFIG } from '../constants.ts';
import { GPU_BUFFER_USAGE_UNIFORM, writeMatrixToGPUBuffer } from '../utils.ts';
import { PassCtx } from './passCtx.ts';

/** Example: https://webgpu.github.io/webgpu-samples/?sample=cameras#cube.wgsl */
export class RenderUniformsBuffer {
  public static BUFFER_SIZE =
    BYTES_MAT4 + // mvpMatrix
    BYTES_MAT4 + // viewMatrix
    BYTES_MAT4 + // projMatrix
    BYTES_VEC4 + // viewportAndFocals
    BYTES_VEC4; // scaleModifier

  private readonly gpuBuffer: GPUBuffer;

  constructor(device: GPUDevice) {
    this.gpuBuffer = device.createBuffer({
      label: 'GlobalUniformBuffer',
      size: RenderUniformsBuffer.BUFFER_SIZE,
      usage: GPU_BUFFER_USAGE_UNIFORM,
    });
  }

  createBindingDesc(bindingIdx: number): GPUBindGroupEntry {
    return {
      binding: bindingIdx,
      resource: {
        buffer: this.gpuBuffer,
      },
    };
  }

  update(ctx: PassCtx) {
    const {
      device,
      mvpMatrix,
      viewMatrix,
      projMatrix,
      viewport,
      focalX,
      focalY,
    } = ctx;
    let offsetBytes = 0;

    writeMatrixToGPUBuffer(device, this.gpuBuffer, offsetBytes, mvpMatrix);
    offsetBytes += BYTES_MAT4;

    writeMatrixToGPUBuffer(device, this.gpuBuffer, offsetBytes, viewMatrix);
    offsetBytes += BYTES_MAT4;

    writeMatrixToGPUBuffer(device, this.gpuBuffer, offsetBytes, projMatrix);
    offsetBytes += BYTES_MAT4;

    // scale as vec4
    const miscF32Array = new Float32Array([
      viewport.width,
      viewport.height,
      focalX,
      focalY,
    ]);
    device.queue.writeBuffer(
      this.gpuBuffer,
      offsetBytes,
      miscF32Array.buffer,
      0
    );
    offsetBytes += BYTES_VEC4;

    // scale as vec4
    miscF32Array[0] = CONFIG.scaleModifier;
    device.queue.writeBuffer(
      this.gpuBuffer,
      offsetBytes,
      miscF32Array.buffer,
      0
    );
    offsetBytes += BYTES_VEC4;
  }
}
