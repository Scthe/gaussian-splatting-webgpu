import { createErrorSystem, createGpuDevice } from './utils.ts';
import { createInputHandler } from './web/input.ts';
import { Renderer, injectShaderTexts } from './renderer.ts';
import { parseFileSplat } from './loaders/fileSplat.ts';
import { initFPSCounter } from './web/fpsStats.ts';
import { initializeGUI, onGpuProfilerResult } from './web/gui.ts';
import { GpuProfiler } from './gpuProfiler.ts';
import { initCanvasResizeSystem } from './web/cavasResize.ts';

//@ts-ignore it works OK
import splatPassShader from './passes/renderSplatsGEO.wgsl';
//@ts-ignore it works OK
import sortBitonicShader from './passes/sortPassGPU/bitonicSort.wgsl';
//@ts-ignore it works OK
import sortCalcDeptsShader from './passes/sortPassGPU/calcDepths.wgsl';
//@ts-ignore it works OK
import sortUnrollIndicesShader from './passes/sortPassGPU/unrollIndices.wgsl';

const SPLAT_FILE = 'nike.splat';

// fix some warnings if VSCode is in deno mode
declare global {
  // deno-lint-ignore no-explicit-any
  function requestAnimationFrame(cb: any): void;
  // deno-lint-ignore no-explicit-any
  type HTMLCanvasElement = any;
  // deno-lint-ignore no-explicit-any
  type CanvasRenderingContext2D = any;
  // deno-lint-ignore no-explicit-any
  const document: any;
}

(async function () {
  // GPUDevice
  const device = await createGpuDevice();
  if (!device) {
    showErrorMessage();
    return;
  }
  const errorSystem = createErrorSystem(device);
  errorSystem.startErrorScope('init');

  // create canvas
  const PREFERRED_CANVAS_FORMAT = navigator.gpu.getPreferredCanvasFormat();
  const [canvas, canvasContext] = getCanvasContext(
    '#gpuCanvas',
    device,
    PREFERRED_CANVAS_FORMAT
  );
  const canvasSize = initCanvasResizeSystem(canvas);

  // input
  const getInputState = createInputHandler(window, canvas);

  // file load
  const splats = await loadScene(device, SPLAT_FILE);

  // renderer setup
  const profiler = new GpuProfiler(device);
  injectShaderTexts({
    splatPassShader,
    sortBitonicShader,
    sortCalcDeptsShader,
    sortUnrollIndicesShader,
  });
  const renderer = new Renderer(
    device,
    canvasSize.getViewportSize(),
    PREFERRED_CANVAS_FORMAT,
    splats
  );
  canvasSize.addListener(renderer.onCanvasResize);

  initializeGUI(profiler);
  const [fpsOnFrameStart, fpsOnFrameEnd] = initFPSCounter();
  let lastFrameMS = Date.now();
  let done = false;

  const lastError = await errorSystem.reportErrorScopeAsync();
  if (lastError) {
    showErrorMessage(lastError);
    return;
  }

  // frame callback
  const frame = () => {
    errorSystem.startErrorScope('frame');

    fpsOnFrameEnd();
    fpsOnFrameStart();
    profiler.beginFrame();
    const now = Date.now();
    const deltaTime = (now - lastFrameMS) / 1000;
    lastFrameMS = now;

    canvasSize.revalidateCanvasSize();

    const inputState = getInputState();
    renderer.updateCamera(deltaTime, inputState);

    // record commands
    const cmdBuf = device.createCommandEncoder({
      label: 'main-frame-cmd-buffer',
    });
    renderer.cmdRender(
      {
        cmdBuf,
        device,
        profiler,
        splats,
        viewport: canvasSize.getViewportSize(),
      },
      canvasContext.getCurrentTexture()
    );

    // submit commands
    profiler.endFrame(cmdBuf);
    device.queue.submit([cmdBuf.finish()]);

    profiler.scheduleRaportIfNeededAsync(onGpuProfilerResult);

    // frame end
    if (!done) {
      errorSystem.reportErrorScopeAsync((lastError) => {
        showErrorMessage(lastError);
        done = true;
        throw new Error(lastError);
      }); // not awaited!

      requestAnimationFrame(frame);
    }
  };

  // start rendering
  requestAnimationFrame(frame);
})();

function getCanvasContext(
  selector: string,
  device: GPUDevice,
  canvasFormat: string
): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.querySelector(selector);
  const context = canvas.getContext('webgpu');

  // const devicePixelRatio = window.devicePixelRatio;
  // canvas.width = canvas.clientWidth * devicePixelRatio;
  // canvas.height = canvas.clientHeight * devicePixelRatio;

  context.configure({
    device,
    format: canvasFormat,
    alphaMode: 'premultiplied',
  });
  return [canvas, context];
}

async function requestBinaryFile(path: string): Promise<ArrayBuffer> {
  const resp = await fetch(path);
  if (!resp.ok) {
    throw `Could not download tfx file '${path}'`;
  }
  return await resp.arrayBuffer();
}

async function loadScene(device: GPUDevice, path: string) {
  const rawBytesArray = await requestBinaryFile(path);
  const rawBytesArrayUints = new Uint8Array(rawBytesArray);
  const splats = parseFileSplat(device, rawBytesArrayUints);
  console.log('Parsed file', splats.count, 'splats');
  return splats;
}

function showErrorMessage(msg?: string) {
  document.getElementById('gpuCanvas').style.display = 'none';
  document.getElementById('no-webgpu').style.display = 'flex';
  if (msg) {
    document.getElementById('error-msg').textContent = msg;
  }
}
