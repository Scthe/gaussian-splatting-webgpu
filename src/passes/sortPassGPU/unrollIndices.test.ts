import { assertEquals } from 'assert';

import { createArray, createGPUBuffer } from '../../utils.ts';
import { UnrollIndicesPass } from './unrollIndices.ts';
import {
  assertGpuDevice,
  cmdCopyToReadBackBuffer,
  createIndicesU32Array,
  createMockPassCtx,
  createReadbackBuffer,
  injectDenoShader,
  readBufferToCPU,
} from '../../deno/testUtils.ts';
import { VERTICES_PER_SPLAT } from '../../gaussianSplats.ts';
import { BYTES_U32 } from '../../constants.ts';
import { unrollSortedSplatIndices } from '../sortPassCPU.ts';

const ITEM_CNT = 325;

Deno.test('Unroll indices', async () => {
  const [device, reportWebGPUErrAsync] = await assertGpuDevice();
  injectDenoShader(UnrollIndicesPass, import.meta.url, 'unrollIndices.wgsl');

  // buffer: original data to unroll
  const indicesData = createIndicesU32Array(ITEM_CNT, true);
  // console.log({ indicesData });
  const indicesBuffer = createGPUBuffer(
    device,
    'unroll-original-indices',
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    indicesData
  );
  assertEquals(indicesBuffer.size, ITEM_CNT * BYTES_U32, indicesBuffer.label);

  // buffer: result
  const unrolledIndicesBuffer = device.createBuffer({
    label: 'unroll-unrolled-indices',
    size: ITEM_CNT * VERTICES_PER_SPLAT * BYTES_U32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  assertEquals(
    unrolledIndicesBuffer.size,
    ITEM_CNT * VERTICES_PER_SPLAT * BYTES_U32,
    unrolledIndicesBuffer.label
  );
  const readbackBuffer = createReadbackBuffer(device, unrolledIndicesBuffer);

  const pass = new UnrollIndicesPass(
    device,
    indicesBuffer,
    unrolledIndicesBuffer,
    ITEM_CNT
  );

  // execute
  const cmdBuf = device.createCommandEncoder();
  const passCtx = createMockPassCtx(device, cmdBuf);
  pass.cmdUnrollIndices(passCtx);
  cmdCopyToReadBackBuffer(cmdBuf, unrolledIndicesBuffer, readbackBuffer);
  device.queue.submit([cmdBuf.finish()]);

  await reportWebGPUErrAsync();

  // assert read back
  const result = await readBufferToCPU(Uint32Array, readbackBuffer);
  // console.log(`Result:`, ...spreadPrintTypedArray(result));

  // assert
  const expected = createArray(ITEM_CNT * VERTICES_PER_SPLAT);
  unrollSortedSplatIndices(indicesData, expected, ITEM_CNT);
  assertEquals(result.length, expected.length);
  for (let i = 0; i < result.length; i++) {
    assertEquals(result[i], expected[i]);
  }

  // cleanup
  device.destroy();
});
