import { calcDepth, createGPUBuffer } from '../../utils.ts';
import {
  assertGpuDevice,
  cmdCopyToReadBackBuffer,
  createMockPassCtx,
  createRandomVec4Array,
  createReadbackBuffer,
  injectDenoShader,
  readBufferToCPU,
} from '../../deno/testUtils.ts';
import { BYTES_U32, BYTES_F32 } from '../../constants.ts';
import { CalcDepthsPass } from './calcDepths.ts';
import { assertAlmostEquals, assertEquals } from 'assert';

const ITEM_CNT = 15;

Deno.test('Calc depths', async () => {
  const [device, reportWebGPUErrAsync] = await assertGpuDevice();
  injectDenoShader(CalcDepthsPass, import.meta.url, 'calcDepths.wgsl');

  // buffer: splat positions
  const positions = createRandomVec4Array(ITEM_CNT, 0);
  // console.log({ positions });
  const splatPositions = createGPUBuffer(
    device,
    'calc-depths-positions',
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    positions
  );

  // buffer: result distances
  const resultDistancesBuffer = device.createBuffer({
    label: 'calc-depths-result-distances',
    size: ITEM_CNT * BYTES_F32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackDistancesBuffer = createReadbackBuffer(
    device,
    resultDistancesBuffer
  );

  // buffer: result indices
  const resultIndicesBuffer = device.createBuffer({
    label: 'calc-depths-result-indices',
    size: ITEM_CNT * BYTES_U32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackIndicesBuffer = createReadbackBuffer(
    device,
    resultIndicesBuffer
  );

  const pass = new CalcDepthsPass(
    device,
    splatPositions,
    resultDistancesBuffer,
    resultIndicesBuffer
  );

  // execute
  const cmdBuf = device.createCommandEncoder();
  const passCtx = createMockPassCtx(device, cmdBuf);
  pass.cmdCalcDepths(passCtx);
  cmdCopyToReadBackBuffer(
    cmdBuf,
    resultDistancesBuffer,
    readbackDistancesBuffer
  );
  cmdCopyToReadBackBuffer(cmdBuf, resultIndicesBuffer, readbackIndicesBuffer);
  device.queue.submit([cmdBuf.finish()]);

  await reportWebGPUErrAsync();

  // assert read back
  const resultDist = await readBufferToCPU(
    Float32Array,
    readbackDistancesBuffer
  );
  const resultIndices = await readBufferToCPU(
    Uint32Array,
    readbackIndicesBuffer
  );
  // console.log(`Result:`, ...spreadPrintTypedArray(resultIndic));

  // assert
  for (let i = 0; i < resultIndices.length; i++) {
    assertEquals(resultIndices[i], i);

    const depth = resultDist[i];
    const expectedDepth = calcDepth(passCtx.mvpMatrix, positions, 4 * i);
    // console.log({
    // expectedDepth,
    // gpu: depth,
    // diff: Math.abs(depth - expectedDepth),
    // });
    assertAlmostEquals(depth, expectedDepth);
  }

  // cleanup
  device.destroy();
});
