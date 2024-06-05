import { BYTES_F32, BYTES_U32 } from '../../constants.ts';
import { GaussianSplats } from '../../gaussianSplats.ts';
import { nearestPowerOf2_ceil } from '../../utils.ts';
import { BitonicSort } from './bitonicSort.ts';
import { CalcDepthsPass } from './calcDepths.ts';
import { UnrollIndicesPass } from './unrollIndices.ts';
import { PassCtx } from '../passCtx.ts';

export class SortPassGPU {
  private readonly itemCountCeilPwr2: number;
  private readonly indicesBuffer: GPUBuffer;
  private readonly distancesBuffer: GPUBuffer;

  private readonly calcDepthsPass: CalcDepthsPass;
  private readonly bitonicSort: BitonicSort;
  private readonly unrollIndicesPass: UnrollIndicesPass;

  constructor(device: GPUDevice, splats: GaussianSplats) {
    this.itemCountCeilPwr2 = nearestPowerOf2_ceil(splats.count);

    const [indicesBuffer, distancesBuffer] = SortPassGPU.createBuffers(
      device,
      this.itemCountCeilPwr2
    );
    this.indicesBuffer = indicesBuffer;
    this.distancesBuffer = distancesBuffer;

    // subpasses
    this.calcDepthsPass = new CalcDepthsPass(
      device,
      splats.positionsBuffer,
      this.distancesBuffer,
      this.indicesBuffer
    );
    this.bitonicSort = new BitonicSort(
      device,
      this.itemCountCeilPwr2,
      this.indicesBuffer,
      this.distancesBuffer
    );
    this.unrollIndicesPass = new UnrollIndicesPass(
      device,
      this.indicesBuffer,
      splats.indicesBuffer,
      splats.count
    );
  }

  private static createBuffers(device: GPUDevice, itemCountCeilPwr2: number) {
    const indicesBuffer = device.createBuffer({
      label: 'sortPassGPU.indices-buffer',
      size: itemCountCeilPwr2 * BYTES_U32,
      usage: GPUBufferUsage.STORAGE,
    });
    const distancesBuffer = device.createBuffer({
      label: 'sortPassGPU.distances-buffer',
      size: itemCountCeilPwr2 * BYTES_F32,
      usage: GPUBufferUsage.STORAGE,
    });
    return [indicesBuffer, distancesBuffer];
  }

  cmdSortByDepth(ctx: PassCtx) {
    // calculate depth-distances and reset indices
    this.calcDepthsPass.cmdCalcDepths(ctx);

    // sort by depth
    this.bitonicSort.cmdSort(ctx);

    // unroll indices to the form the renderer expects
    this.unrollIndicesPass.cmdUnrollIndices(ctx);
  }
}
