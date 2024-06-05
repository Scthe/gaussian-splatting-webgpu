import * as path from 'std-path';
import {
  Dimensions,
  createArray,
  createCameraProjectionMat,
  createErrorSystem,
  createGpuDevice,
  getClassName,
  getModelViewProjectionMatrix,
  rethrowWebGPUError,
} from '../utils.ts';
import { PassCtx } from '../passes/passCtx.ts';
import { Camera2 } from '../web/camera2.ts';

export function injectDenoShader(
  PassClass: { SHADER_CODE: string },
  modulePath: string, // import.meta.url
  shaderFile: string
) {
  const __dirname = path.dirname(path.fromFileUrl(modulePath));
  PassClass.SHADER_CODE = Deno.readTextFileSync(
    `${__dirname}${path.SEPARATOR}${shaderFile}`
  );
}

type ValidateWebGPUCallsFn = () => Promise<void>;

export async function assertGpuDevice(): Promise<
  [GPUDevice, ValidateWebGPUCallsFn]
> {
  const device = await createGpuDevice();
  if (!device) throw new Error('No webgpu device');

  const errorSystem = createErrorSystem(device);
  errorSystem.startErrorScope();
  const validateFn = async () => {
    await errorSystem.reportErrorScopeAsync(rethrowWebGPUError);
  };

  return [device, validateFn];
}

export const createMockPassCtx = (
  device: GPUDevice,
  cmdBuf: GPUCommandEncoder
): PassCtx => {
  const viewport: Dimensions = {
    width: 800,
    height: 600,
  };

  const cameraCtrl = new Camera2();
  const projMatrix = createCameraProjectionMat(viewport);
  const mvpMatrix = getModelViewProjectionMatrix(
    cameraCtrl.viewMatrix,
    projMatrix
  );

  return {
    device,
    cmdBuf,
    mvpMatrix,
    viewMatrix: cameraCtrl.viewMatrix,
    projMatrix,
    profiler: undefined,
    viewport,
    focalX: 600,
    focalY: 600,
    splats: undefined!,
  };
};

// deno-lint-ignore no-explicit-any
export async function readBufferToCPU(TypedArrayClass: any, buffer: GPUBuffer) {
  await buffer.mapAsync(1);
  const arrayBufferData = buffer.getMappedRange();
  const resultData = new TypedArrayClass(arrayBufferData);
  buffer.unmap();
  return resultData;
}

export function spreadPrintTypedArray(
  arr: Float32Array | Uint32Array,
  cnt = 0
) {
  cnt = cnt > 0 ? cnt : arr.length;
  const a = Array.from(arr).slice(0, cnt);
  const typeName = getClassName(arr);
  return [`${typeName}(len=${arr.length}, bytes=${arr.byteLength})`, a];
}

///////////////
/// Readback GPU->CPU

export function createReadbackBuffer(device: GPUDevice, orgBuffer: GPUBuffer) {
  return device.createBuffer({
    label: 'test-readback',
    size: orgBuffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
}

export function cmdCopyToReadBackBuffer(
  cmdBuf: GPUCommandEncoder,
  orgBuffer: GPUBuffer,
  readbackBuffer: GPUBuffer
) {
  cmdBuf.copyBufferToBuffer(orgBuffer, 0, readbackBuffer, 0, orgBuffer.size);
}

///////////////
/// Create mock data

export function createIndicesU32Array(cnt: number, shuffle = false) {
  const arr = createArray(cnt).map((_, idx) => idx);

  if (shuffle) {
    shuffleArray(arr);
  }

  const result = new Uint32Array(cnt);
  arr.forEach((e, idx) => (result[idx] = e));
  return result;
}

export function createRandomF32Array(cnt: number) {
  const result = new Float32Array(cnt);
  for (let i = 0; i < cnt; i++) {
    result[i] = Math.random();
  }
  return result;
}

export function createRandomVec4Array(cnt: number, w = 1) {
  const rngCoord = () => Math.random() * 2 - 1;
  const result = new Float32Array(cnt * 4);
  for (let i = 0; i < cnt; i++) {
    result[i * 4] = rngCoord();
    result[i * 4 + 1] = rngCoord();
    result[i * 4 + 2] = rngCoord();
    result[i * 4 + 3] = w;
  }
  return result;
}

function shuffleArray(array: number[]) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}