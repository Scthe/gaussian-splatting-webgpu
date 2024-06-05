import { Mat4, mat4 } from 'wgpu-matrix';
import { CAMERA_CFG } from './constants.ts';

export interface Dimensions {
  width: number;
  height: number;
}

export type TypedArray = Float32Array | Uint8Array | Uint32Array;

export async function createGpuDevice() {
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    const onError = (msg: string) =>
      console.error(`WebGPU init error: '${msg}'`);

    if (!adapter) {
      // On web: check if https. On ff, WebGPU is under dev flag.
      onError('No adapter found. WebGPU seems to be unavailable.');
      return;
    }

    const canTimestamp = adapter.features.has('timestamp-query');
    const requiredFeatures: GPUFeatureName[] = [];
    if (canTimestamp) {
      requiredFeatures.push('timestamp-query');
    }

    const device = await adapter?.requestDevice({ requiredFeatures });
    if (!device) {
      onError('Failed to get GPUDevice from the adapter.');
      return;
    }

    return device;
  } catch (e) {
    console.error(e);
    return;
  }
}

export const dgr2rad = (dgr: number) => (dgr * Math.PI) / 180;

export function createCameraProjectionMat(viewportSize: Dimensions): Mat4 {
  const aspectRatio = viewportSize.width / viewportSize.height;

  return mat4.perspective(
    dgr2rad(CAMERA_CFG.fovDgr),
    aspectRatio,
    CAMERA_CFG.near,
    CAMERA_CFG.far
  );
}

export function getModelViewProjectionMatrix(
  viewMat: Mat4,
  projMat: Mat4
): Mat4 {
  return mat4.multiply(projMat, viewMat);
}

/** debug matrix to string */
export function dbgMat(m: Float32Array) {
  const s = Math.sqrt(m.length);
  let result = '';
  for (let i = 0; i < m.length; i++) {
    if (i % s === 0) result += '\n';
    else result += '   ';
    result += m[i].toFixed(2);
  }
  return `[${result}\n]`;
}

export const GPU_BUFFER_USAGE_UNIFORM: GPUBufferUsageFlags =
  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export function createGPUBuffer<T extends TypedArray>(
  device: GPUDevice,
  label: string,
  usage: GPUBufferUsageFlags,
  data: T
) {
  const gpuBuffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage,
  });
  // device.queue.writeBuffer(gpuBuffer, 0, data, 0, data.length);
  device.queue.writeBuffer(gpuBuffer, 0, data);
  return gpuBuffer;
}

export function writeMatrixToGPUBuffer(
  device: GPUDevice,
  gpuBuffer: GPUBuffer,
  offsetBytes: number,
  data: Mat4
) {
  // deno-lint-ignore no-explicit-any
  const f32Arr: Float32Array = data as any;
  device.queue.writeBuffer(gpuBuffer, offsetBytes, f32Arr.buffer, 0);
}

export function makeUint(x: number, y: number, z: number, w: number) {
  const a = (x & 0xff) << 24;
  const b = (y & 0xff) << 16;
  const c = (z & 0xff) << 8;
  const d = (w & 0xff) << 0;
  return a + b + c + d;
}

/** Returns floor. Examples:
 * - for 3 returns 2
 * - for 4 returns 4
 * - for 7 returns 4
 * - for 8 returns 8
 */
export function nearestPowerOf2_floor(n: number) {
  return 1 << (31 - Math.clz32(n));
}

export function nearestPowerOf2_ceil(n: number) {
  const floor = nearestPowerOf2_floor(n);
  return floor === n ? n : 2 * floor;
}

export function getClassName(a: object) {
  // deno-lint-ignore no-explicit-any
  return (a as any).constructor.name;
}

export const createArray = (len: number) => Array(len).fill(0);

/** Multiplies vec3 (`pos.w` is ignored). */
export function calcDepth(
  mvp: Mat4,
  pointsVec4: Float32Array,
  pointsOffset: number
) {
  return (
    mvp[2] * pointsVec4[pointsOffset] +
    mvp[6] * pointsVec4[pointsOffset + 1] +
    mvp[10] * pointsVec4[pointsOffset + 2]
  );
}

export function assertHasInjectedShader(clazz: {
  SHADER_CODE: string;
  name: string;
}) {
  if (!clazz.SHADER_CODE || clazz.SHADER_CODE.length == 0) {
    throw new Error(`${clazz.name} has no .SHADER_CODE defined.`);
  }
}

type ShaderOverrides = { [key: string]: string };

/**
 * In WGSL there is something called overrides:
 *  - https://www.w3.org/TR/WGSL/#override-declaration
 *  - https://webgpufundamentals.org/webgpu/lessons/webgpu-constants.html
 * Would have been better than text replace. But neither works
 * with language servers in text editors, so might as well text replace.
 */
export function applyShaderTextReplace(
  text: string,
  overrides?: ShaderOverrides
) {
  let code = text;
  overrides = overrides || {};
  Object.entries(overrides).forEach(([k, v]) => {
    code = code.replaceAll(k, v);
  });
  return code;
}

export const getItemsPerThread = (items: number, threads: number) =>
  Math.ceil(items / threads);

export const lerp = (a: number, b: number, fac: number) => {
  fac = Math.max(0, Math.min(1, fac));
  return a * (1 - fac) + b * fac;
};

type ErrorCb = (msg: string) => never;

export function createErrorSystem(device: GPUDevice) {
  const ERROR_SCOPES: GPUErrorFilter[] = [
    'internal',
    'out-of-memory',
    'validation',
  ];
  const ERROR_SCOPES_REV = ERROR_SCOPES.toReversed();

  let currentScopeName = '-';

  return {
    startErrorScope,
    reportErrorScopeAsync,
  };

  function startErrorScope(scopeName: string = '-') {
    currentScopeName = scopeName;
    ERROR_SCOPES.forEach((sc) => device.pushErrorScope(sc));
  }

  async function reportErrorScopeAsync(cb?: ErrorCb) {
    let lastError = undefined;

    for (const name of ERROR_SCOPES_REV) {
      const err = await device.popErrorScope();
      if (err) {
        const msg = `WebGPU error [${currentScopeName}][${name}]: ${err.message}`;
        lastError = msg;
        if (cb) {
          cb(msg);
        } else {
          console.error(msg);
        }
      }
    }

    return lastError;
  }
}

export const rethrowWebGPUError = (msg: string): never => {
  throw new Error(msg);
};
