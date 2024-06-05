import { GaussianSplats, VERTICES_PER_SPLAT } from '../gaussianSplats.ts';
import { calcDepth, createArray } from '../utils.ts';
import {
  unrollSortedSplatIndices,
  writeUnrolledIndicesToGpu,
} from './sortPassCPU.ts';
import { PassCtx } from './passCtx.ts';

interface Item {
  depth: number;
  idx: number;
}

export class SortPassCPU_Naive {
  private static NAME = SortPassCPU_Naive.name;

  private readonly arrayToSort: Item[];
  private readonly sortedSplatsIndices: Uint32Array;
  private readonly unroledSortedSplatsIndices: Uint32Array;

  constructor(splats: GaussianSplats) {
    const splatCount = splats.count;
    this.arrayToSort = createArray(splatCount).map((_) => ({
      depth: 0,
      idx: 0,
    }));
    this.sortedSplatsIndices = new Uint32Array(splatCount);
    this.unroledSortedSplatsIndices = new Uint32Array(
      splatCount * VERTICES_PER_SPLAT
    );
  }

  cmdSortByDepth(ctx: PassCtx) {
    const { device, mvpMatrix, splats, profiler } = ctx;
    const profilerScope = profiler?.startRegionCpu(SortPassCPU_Naive.NAME);

    // recalc depts
    this.arrayToSort.forEach((e, idx) => {
      e.idx = idx;
      e.depth = calcDepth(mvpMatrix, splats.positions, 4 * idx);
    });

    // naive JS sort
    this.arrayToSort.sort((a, b) => {
      return b.depth - a.depth;
    });

    // flatten so we can reuse utils
    this.arrayToSort.forEach((e, idx) => {
      this.sortedSplatsIndices[idx] = e.idx;
    });

    unrollSortedSplatIndices(
      this.sortedSplatsIndices,
      this.unroledSortedSplatsIndices,
      splats.count
    );

    profiler?.endRegionCpu(profilerScope);

    writeUnrolledIndicesToGpu(device, splats, this.unroledSortedSplatsIndices);
  }
}
