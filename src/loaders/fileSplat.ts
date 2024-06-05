import { BYTES_F32, BYTES_U32, BYTES_U8 } from '../constants.ts';
import {
  GaussianSplats,
  SPLAT_SIZE_BYTES,
  VERTICES_PER_SPLAT,
} from '../gaussianSplats.ts';
import { makeUint, createGPUBuffer } from '../utils.ts';

const OFFSET_SCALE = 3 * BYTES_F32;
const OFFSET_COLOR = OFFSET_SCALE + 3 * BYTES_F32;
const OFFSET_ROTATION = OFFSET_COLOR + 4 * BYTES_U8;
const offsetBytesToOffsetF32 = (e: number) => e / BYTES_F32;

/**
 * https://github.com/antimatter15/splat/blob/main/main.js#L1391
 */
export function parseFileSplat(
  device: GPUDevice,
  data: Uint8Array
): GaussianSplats {
  const dataBytesCnt = data.length;
  const splatCount = dataBytesCnt / SPLAT_SIZE_BYTES;
  console.log(`bytes: ${data.length} splats: ${splatCount}`);

  // careful to REINTERPRET the data, instead of element-wise cast!
  const dataF32 = new Float32Array(data.buffer);
  if (dataF32.length !== data.length / 4) {
    throw new Error(
      `File bytes (${data.length}) could not be reinterpreted as floats`
    );
  }

  const positions = new Float32Array(splatCount * 4);
  const scales = new Float32Array(splatCount * 4);
  const rotations = new Float32Array(splatCount * 4);
  const colors = new Uint32Array(splatCount);

  for (let i = 0; i < splatCount; i++) {
    const offsetBytes = i * SPLAT_SIZE_BYTES;
    const offsetBytesIntoF32 = offsetBytesToOffsetF32(offsetBytes);

    positions[i * 4 + 0] = dataF32[offsetBytesIntoF32];
    positions[i * 4 + 1] = dataF32[offsetBytesIntoF32 + 1];
    positions[i * 4 + 2] = dataF32[offsetBytesIntoF32 + 2];
    positions[i * 4 + 3] = 1; // IMPORTANT!
    const offsetScaleF32 = offsetBytesToOffsetF32(OFFSET_SCALE);
    scales[i * 4 + 0] = dataF32[offsetBytesIntoF32 + offsetScaleF32];
    scales[i * 4 + 1] = dataF32[offsetBytesIntoF32 + offsetScaleF32 + 1];
    scales[i * 4 + 2] = dataF32[offsetBytesIntoF32 + offsetScaleF32 + 2];
    scales[i * 4 + 3] = 1;
    const r = (x: number) => (x - 128.0) / 128.0;
    rotations[i * 4 + 0] = r(data[offsetBytes + OFFSET_ROTATION]);
    rotations[i * 4 + 1] = r(data[offsetBytes + OFFSET_ROTATION + 1]);
    rotations[i * 4 + 2] = r(data[offsetBytes + OFFSET_ROTATION + 2]);
    rotations[i * 4 + 3] = r(data[offsetBytes + OFFSET_ROTATION + 3]);

    colors[i] = makeUint(
      data[offsetBytes + OFFSET_COLOR],
      data[offsetBytes + OFFSET_COLOR + 1],
      data[offsetBytes + OFFSET_COLOR + 2],
      data[offsetBytes + OFFSET_COLOR + 3]
    );
  }

  printPositionsStats(positions);

  const usage: GPUBufferUsageFlags =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const positionsBuffer = createGPUBuffer(
    device,
    'splats-positions',
    usage,
    positions
  );
  const rotationsBuffer = createGPUBuffer(
    device,
    'splats-rotations',
    usage,
    rotations
  );
  const scalesBuffer = createGPUBuffer(device, 'splats-scales', usage, scales);
  const colorsBuffer = createGPUBuffer(device, 'splats-colors', usage, colors);

  const indicesBuffer = device.createBuffer({
    label: 'splats-indices',
    size: splatCount * VERTICES_PER_SPLAT * BYTES_U32,
    usage:
      GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return {
    count: splatCount,
    positions,
    positionsBuffer,
    colorsBuffer,
    rotationsBuffer,
    scalesBuffer,
    indicesBuffer,
  };
}

function printPositionsStats(data: Float32Array) {
  const maxCo = [data[0], data[1], data[2]];
  const minCo = [data[0], data[1], data[2]];
  const splatCount = data.length / 4;

  for (let i = 0; i < splatCount; i++) {
    const offset = i * 4;
    for (let co = 0; co < 3; co++) {
      maxCo[co] = Math.max(maxCo[co], data[offset + co]);
      minCo[co] = Math.min(minCo[co], data[offset + co]);
    }
  }

  const p = (a: number[]) => '[' + a.map((x) => x.toFixed(2)).join(',') + ']';
  console.log(`Bounding box min:`, p(minCo));
  console.log(`Bounding box max:`, p(maxCo));
}
