import { VERTICES_PER_SPLAT } from '../../gaussianSplats.ts';
import {
  applyShaderTextReplace,
  assertHasInjectedShader,
  getItemsPerThread,
} from '../../utils.ts';
import { PassCtx } from '../passCtx.ts';

export class UnrollIndicesPass {
  public static NAME: string = UnrollIndicesPass.name;
  private static readonly NUM_THREADS = 64;
  public static SHADER_CODE: string = '';

  private readonly pipeline: GPUComputePipeline;
  private readonly uniformsBindings: GPUBindGroup;

  constructor(
    device: GPUDevice,
    indicesBuffer: GPUBuffer,
    unrolledIndicesBuffer: GPUBuffer,
    splatCount: number
  ) {
    assertHasInjectedShader(UnrollIndicesPass);
    const itemsPerThread = getItemsPerThread(
      splatCount,
      UnrollIndicesPass.NUM_THREADS
    );

    this.pipeline = UnrollIndicesPass.createPipeline(device, itemsPerThread);

    this.uniformsBindings = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: indicesBuffer },
        },
        {
          binding: 1,
          resource: { buffer: unrolledIndicesBuffer },
        },
      ],
    });
  }

  private static createPipeline(device: GPUDevice, itemsPerThread: number) {
    const code = applyShaderTextReplace(UnrollIndicesPass.SHADER_CODE, {
      __ITEMS_PER_THREAD__: '' + itemsPerThread,
      __VERTICES_PER_SPLAT__: '' + VERTICES_PER_SPLAT,
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

  cmdUnrollIndices(ctx: PassCtx) {
    const { cmdBuf, profiler } = ctx;

    const computePass = cmdBuf.beginComputePass({
      label: 'unroll-indices-pass',
      timestampWrites: profiler?.createScopeGpu(UnrollIndicesPass.NAME),
    });
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, this.uniformsBindings);
    computePass.dispatchWorkgroups(UnrollIndicesPass.NUM_THREADS);
    computePass.end();
  }
}
