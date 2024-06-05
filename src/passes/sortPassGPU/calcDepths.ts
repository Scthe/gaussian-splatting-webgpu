import { BYTES_VEC4, BYTES_MAT4 } from '../../constants.ts';
import {
  GPU_BUFFER_USAGE_UNIFORM,
  applyShaderTextReplace,
  assertHasInjectedShader,
  getItemsPerThread,
  writeMatrixToGPUBuffer,
} from '../../utils.ts';
import { PassCtx } from '../passCtx.ts';

export class CalcDepthsPass {
  public static NAME: string = CalcDepthsPass.name;
  private static readonly NUM_THREADS = 64;
  public static SHADER_CODE: string = '';

  private readonly pipeline: GPUComputePipeline;
  private readonly uniformsBindings: GPUBindGroup;
  private readonly uniformsBuffer: GPUBuffer;

  constructor(
    device: GPUDevice,
    splatPositions: GPUBuffer,
    // F32 buffer of size nearestPowerOf2_ceil($splatCount)
    distancesBuffer: GPUBuffer,
    // U32 buffer of size nearestPowerOf2_ceil($splatCount)
    indicesBuffer: GPUBuffer
  ) {
    assertHasInjectedShader(CalcDepthsPass);
    const splatCount = splatPositions.size / BYTES_VEC4;
    const itemsPerThread = getItemsPerThread(
      splatCount,
      CalcDepthsPass.NUM_THREADS
    );

    this.uniformsBuffer = device.createBuffer({
      label: 'CalcDepthsPass-uniforms',
      size: BYTES_MAT4,
      usage: GPU_BUFFER_USAGE_UNIFORM,
    });

    this.pipeline = CalcDepthsPass.createPipeline(
      device,
      splatCount,
      itemsPerThread
    );

    this.uniformsBindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: splatPositions },
        },
        {
          binding: 1,
          resource: { buffer: distancesBuffer },
        },
        {
          binding: 2,
          resource: { buffer: indicesBuffer },
        },
        {
          binding: 3,
          resource: { buffer: this.uniformsBuffer },
        },
      ],
    });
  }

  private static createPipeline(
    device: GPUDevice,
    splatCount: number,
    itemsPerThread: number
  ) {
    const code = applyShaderTextReplace(CalcDepthsPass.SHADER_CODE, {
      __ITEMS_PER_THREAD__: '' + itemsPerThread,
      __SPLAT_COUNT__: '' + splatCount,
    });
    const shaderModule = device.createShaderModule({ code });
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  cmdCalcDepths(ctx: PassCtx) {
    const { cmdBuf, device, mvpMatrix, profiler } = ctx;

    writeMatrixToGPUBuffer(device, this.uniformsBuffer, 0, mvpMatrix);

    const computePass = cmdBuf.beginComputePass({
      label: 'calc-depths-pass',
      timestampWrites: profiler?.createScopeGpu(CalcDepthsPass.NAME),
    });
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, this.uniformsBindings);
    computePass.dispatchWorkgroups(CalcDepthsPass.NUM_THREADS);
    computePass.end();
  }
}
