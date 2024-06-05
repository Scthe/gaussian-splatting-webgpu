import { BitonicSort } from './bitonicSort.ts';
import {
  assertGpuDevice,
  cmdCopyToReadBackBuffer,
  createIndicesU32Array,
  createMockPassCtx,
  createRandomF32Array,
  createReadbackBuffer,
  injectDenoShader,
  readBufferToCPU,
} from '../../deno/testUtils.ts';
import { createGPUBuffer, nearestPowerOf2_ceil } from '../../utils.ts';
import { assert, assertAlmostEquals } from 'assert';

const ITEM_CNT = (1 << 10) + 30;
// const ITEM_CNT = 64;
const ARRAY_LENGTH = nearestPowerOf2_ceil(ITEM_CNT);

Deno.test('Bitonic sort', async () => {
  const [device, reportWebGPUErrAsync] = await assertGpuDevice();
  injectDenoShader(BitonicSort, import.meta.url, 'bitonicSort.wgsl');

  // buffer: distances
  const distancesData = createRandomF32Array(ARRAY_LENGTH);
  // console.log('distancesData', distancesData);
  const distancesBuffer = createGPUBuffer(
    device,
    'bitonic-sort-distances',
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    distancesData
  );
  const readbackDistancesBuffer = createReadbackBuffer(device, distancesBuffer);

  // buffer: indices
  const indicesData = createIndicesU32Array(ARRAY_LENGTH);
  // console.log('indicesData', indicesData);
  const indicesBuffer = createGPUBuffer(
    device,
    'bitonic-sort-indices',
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    indicesData
  );
  const readbackIndicesBuffer = createReadbackBuffer(device, indicesBuffer);

  // pass
  const pass = new BitonicSort(
    device,
    ARRAY_LENGTH,
    indicesBuffer,
    distancesBuffer
  );

  // submit
  const cmdBuf = device.createCommandEncoder();
  const passCtx = createMockPassCtx(device, cmdBuf);
  pass.cmdSort(passCtx);
  cmdCopyToReadBackBuffer(cmdBuf, distancesBuffer, readbackDistancesBuffer);
  cmdCopyToReadBackBuffer(cmdBuf, indicesBuffer, readbackIndicesBuffer);
  device.queue.submit([cmdBuf.finish()]);

  await reportWebGPUErrAsync();

  // read back
  const resultDistances = await readBufferToCPU(
    Float32Array,
    readbackDistancesBuffer
  );
  const resultIndices = await readBufferToCPU(
    Uint32Array,
    readbackIndicesBuffer
  );
  // console.log(`resultDistances:`, ...spreadPrintTypedArray(resultDistances));
  // console.log(`resultIndices:`, ...spreadPrintTypedArray(resultIndices));

  for (let i = 0; i < ITEM_CNT; i++) {
    const idx = resultIndices[i];
    const depth = resultDistances[i];

    // check the proper distance for this index
    assertAlmostEquals(depth, distancesData[idx]);

    // check is sorted
    if (i == 0) continue;
    const prevDepth = resultDistances[i - 1];
    assert(prevDepth <= depth);
  }

  // cleanup
  device.destroy();
});
