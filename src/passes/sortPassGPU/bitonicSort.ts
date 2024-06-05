import {
  GPU_BUFFER_USAGE_UNIFORM,
  applyShaderTextReplace,
  assertHasInjectedShader,
  createGPUBuffer,
  getItemsPerThread,
} from '../../utils.ts';
import { PassCtx } from '../passCtx.ts';

/** https://en.wikipedia.org/wiki/Bitonic_sorter */
export class BitonicSort {
  public static NAME: string = BitonicSort.name;
  public static SHADER_CODE: string = '';
  private static NUM_THREADS = 8192;
  private static WORKGROUP_SIZE = 128;

  private readonly pipeline: GPUComputePipeline;
  private readonly gpuUniformBuffers: GPUBuffer[];
  private readonly gpuUniformBuffersBindGroups: GPUBindGroup[];

  constructor(
    device: GPUDevice,
    itemCountCeilPwr2: number,
    indicesBuffer: GPUBuffer,
    distancesBuffer: GPUBuffer
  ) {
    assertHasInjectedShader(BitonicSort);
    const itemsPerThread = getItemsPerThread(
      itemCountCeilPwr2,
      BitonicSort.NUM_THREADS
    );

    this.pipeline = BitonicSort.createPipeline(device, itemsPerThread);

    this.gpuUniformBuffers = BitonicSort.createUniformBuffers(
      device,
      itemCountCeilPwr2
    );
    console.log(
      `Bitonic sort will have ${this.gpuUniformBuffers.length} passes.`
    );
    this.gpuUniformBuffersBindGroups = this.gpuUniformBuffers.map(
      (uniformBuffer) =>
        BitonicSort.createBindGroup(
          device,
          this.pipeline,
          indicesBuffer,
          distancesBuffer,
          uniformBuffer
        )
    );
  }

  private static createPipeline(device: GPUDevice, itemsPerThread: number) {
    const code = applyShaderTextReplace(BitonicSort.SHADER_CODE, {
      __ITEMS_PER_THREAD__: '' + itemsPerThread,
      __WORKGROUP_SIZE__: '' + BitonicSort.WORKGROUP_SIZE,
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

  cmdSort(ctx: PassCtx) {
    const { cmdBuf, profiler } = ctx;

    /*const profilerScope = this.profiler.startRegionGpu(
      cmdBuf,
      `${BitonicSort.NAME}(${this.gpuUniformBuffersBindGroups.length} calls)`
    );*/

    this.gpuUniformBuffersBindGroups.forEach((uniformBindGroup) => {
      const computePass = cmdBuf.beginComputePass({
        timestampWrites: profiler?.createScopeGpu(BitonicSort.NAME),
      });
      computePass.setPipeline(this.pipeline);
      computePass.setBindGroup(0, uniformBindGroup);
      computePass.dispatchWorkgroups(
        BitonicSort.NUM_THREADS / BitonicSort.WORKGROUP_SIZE
      );
      computePass.end();
    });

    // this.profiler.endRegionGpu(cmdBuf, profilerScope);
  }

  /**
   * TODO inline k,j iterations into the kernel? No global barriers means fancy with workgroup barriers?
   *
   * See below for better, but not working solutions.
   * TL;DR: Uniforms have to be aligned to 256 bytes, we only have BYTES_U32*2=8.
   * That would be a massive waste of space.
   */
  private static createUniformBuffers(device: GPUDevice, elementCount: number) {
    const uniformBuffers = [];

    // WIKIPEDIA: k is doubled every iteration
    for (let k = 2; k <= elementCount; k <<= 1) {
      // WIKIPEDIA: j is halved at every iteration, with truncation of fractional parts
      //
      // since JS is.. JS, i'd rather bit shift instead of divide.
      // Call me paranoid..
      for (let j = k >> 1; j > 0; j >>= 1) {
        const bufferContent = new Uint32Array([j, k]);
        // console.log('BitonicSort:', { j, k });

        const gpuBuffer = createGPUBuffer(
          device,
          `bitonic-sort.uniforms-buffer(k=${k},j=${j})`,
          GPU_BUFFER_USAGE_UNIFORM,
          bufferContent
        );
        uniformBuffers.push(gpuBuffer);
      }
    }

    return uniformBuffers;
  }

  private static createBindGroup = (
    device: GPUDevice,
    computePipeline: GPUComputePipeline,
    indicesBuffer: GPUBuffer,
    distancesBuffer: GPUBuffer,
    uniformsBuffer: GPUBuffer
  ) =>
    device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: indicesBuffer },
        },
        {
          binding: 1,
          resource: { buffer: distancesBuffer },
        },
        {
          binding: 2,
          resource: { buffer: uniformsBuffer },
        },
      ],
    });

  /*
  Alternatives to uniforms:
  1) 1 GPUBuffer with uniforms, 1 GpuBindGroup controlled by offsets
     Uses `computePass.setBindGroup(0, gpuBindGroup, [byteOffset]);`
     Does not work cause:
      - "Dynamic Offset[0] (8) is not 256 byte aligned."
  2) 1 GPUBuffer with uniforms, many GpuBindGroups
     Each bind group specifies GPUBuffer subset using: 
      - `resource: { buffer, offset: bufferOffsetBytes, size: itemSize },`
      Does not work cause:
        - "does not satisfy the minimum BufferBindingType::Uniform alignment (256)."


  private static createUniformsOriginal(
    device: GPUDevice,
    computePipeline: GPUComputePipeline,
    indicesBuffer: GPUBuffer,
    distancesBuffer: GPUBuffer,
    elementCount: number
  ): [GPUBuffer, Array<GPUBindGroup>] {
    const jkArray: Array<[number, number]> = [];

    // WIKIPEDIA: k is doubled every iteration
    for (let k = 2; k <= elementCount; k <<= 1) {
      // WIKIPEDIA: j is halved at every iteration, with truncation of fractional parts
      //
      // since JS is.. JS, i'd rather bit shift instead of divide.
      // Call me paranoid..
      for (let j = k >> 1; j > 0; j >>= 1) {
        jkArray.push([j, k]);
      }
    }
    console.log(
      `Bitonic sort will have ${jkArray.length} passes. Array<[j,k]>:`,
      jkArray
    );

    // single unified buffer for all
    const bufferContent = new Uint32Array(jkArray.length * 2);
    jkArray.forEach(([j, k], idx) => {
      bufferContent[2 * idx] = j;
      bufferContent[2 * idx + 1] = k;
    });
    const buffer = createGPUBuffer(
      device,
      `bitonic-sort.uniforms-buffer`,
      GPU_BUFFER_USAGE_UNIFORM,
      bufferContent
    );

    const itemSize = 2 * BYTES_U32;
    const bindGroups = jkArray.map((_, idx) => {
      const bufferOffsetBytes = idx * itemSize;
      return device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: { buffer: indicesBuffer },
          },
          {
            binding: 1,
            resource: {
              buffer: distancesBuffer,
            },
          },
          {
            binding: 2,
            resource: { buffer, offset: bufferOffsetBytes, size: itemSize },
          },
        ],
      });
    });

    return [buffer, bindGroups];
  }
  */
}
