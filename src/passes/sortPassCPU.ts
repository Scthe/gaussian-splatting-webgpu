import { Mat4 } from 'wgpu-matrix';
import { GaussianSplats, VERTICES_PER_SPLAT } from '../gaussianSplats.ts';
import { calcDepth } from '../utils.ts';
import { PassCtx } from './passCtx.ts';

const DEPTH_MUL = 4096;
const SORTING_BINS_COUNT = 256 * 256;

export class SortPassCPU {
  private static NAME = SortPassCPU.name;

  private readonly sizeList: Int32Array;
  private readonly starts0: Uint32Array;
  private readonly sortedSplatsIndices: Uint32Array;
  private readonly unroledSortedSplatsIndices: Uint32Array;

  constructor(splats: GaussianSplats) {
    const splatCount = splats.count;
    this.sizeList = new Int32Array(splatCount);
    this.starts0 = new Uint32Array(SORTING_BINS_COUNT);
    this.sortedSplatsIndices = new Uint32Array(splatCount);
    this.unroledSortedSplatsIndices = new Uint32Array(
      splatCount * VERTICES_PER_SPLAT
    );
  }

  cmdSortByDepth(ctx: PassCtx) {
    const { device, mvpMatrix, splats, profiler } = ctx;

    const profilerScope = profiler?.startRegionCpu(SortPassCPU.NAME);
    const splatCount = splats.count;
    const [minDepth, maxDepth] = this.calculateDepths(mvpMatrix, splats);

    const depthPerBin = (maxDepth - minDepth) / SORTING_BINS_COUNT;
    const splatCountPerBin = new Uint32Array(SORTING_BINS_COUNT);
    for (let i = 0; i < splatCount; i++) {
      const binIdx = ((this.sizeList[i] - minDepth) / depthPerBin) | 0;
      this.sizeList[i] = binIdx;
      splatCountPerBin[binIdx]++;
    }

    const starts0 = this.starts0;
    for (let i = 1; i < SORTING_BINS_COUNT; i++) {
      starts0[i] = starts0[i - 1] + splatCountPerBin[i - 1];
    }

    for (let i = 0; i < splatCount; i++) {
      this.sortedSplatsIndices[starts0[this.sizeList[i]]++] = i;
    }

    unrollSortedSplatIndices(
      this.sortedSplatsIndices,
      this.unroledSortedSplatsIndices,
      splats.count
    );

    profiler?.endRegionCpu(profilerScope);

    writeUnrolledIndicesToGpu(device, splats, this.unroledSortedSplatsIndices);
  }

  private calculateDepths(mvpMatrix: Mat4, splats: GaussianSplats) {
    let maxDepth = -Infinity;
    let minDepth = Infinity;
    for (let i = 0; i < splats.count; i++) {
      let depth = calcDepth(mvpMatrix, splats.positions, 4 * i);
      depth = Math.floor(depth * DEPTH_MUL);
      this.sizeList[i] = depth; // converts to int
      maxDepth = Math.max(maxDepth, depth);
      minDepth = Math.min(minDepth, depth);
    }
    return [minDepth, maxDepth];
  }
}

export function unrollSortedSplatIndices(
  indices: Uint32Array | number[],
  unrolledIndicesArray: Uint32Array | number[],
  splatCount: number
) {
  if (splatCount !== indices.length) {
    throw new Error(
      `There are ${splatCount} splats, but there are ${indices.length} splat sorted indices. This number should match.`
    );
  }

  for (let i = 0; i < splatCount; i++) {
    const idx = indices[i];
    for (let j = 0; j < VERTICES_PER_SPLAT; j++) {
      // content has to be [0,1,2,3,4,5] for each of the splats
      unrolledIndicesArray[i * VERTICES_PER_SPLAT + j] =
        idx * VERTICES_PER_SPLAT + j;
    }
  }
}

export function writeUnrolledIndicesToGpu(
  device: GPUDevice,
  splats: GaussianSplats,
  unrolledIndicesArray: Uint32Array
) {
  device.queue.writeBuffer(splats.indicesBuffer, 0, unrolledIndicesArray, 0);
}
