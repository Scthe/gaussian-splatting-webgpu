import { Mat4 } from 'wgpu-matrix';

import { RenderUniformsBuffer } from './passes/renderUniformsBuffer.ts';
import { RenderSplatsGEO } from './passes/renderSplatsGEO.ts';
import {
  Dimensions,
  createCameraProjectionMat,
  getModelViewProjectionMatrix,
} from './utils.ts';
import Input from './web/input.ts';
import { GaussianSplats } from './gaussianSplats.ts';
import { CAMERA_CFG, CONFIG } from './constants.ts';
import { SortPassGPU } from './passes/sortPassGPU/index.ts';
import { BitonicSort } from './passes/sortPassGPU/bitonicSort.ts';
import { CalcDepthsPass } from './passes/sortPassGPU/calcDepths.ts';
import { UnrollIndicesPass } from './passes/sortPassGPU/unrollIndices.ts';
import { SortPassCPU } from './passes/sortPassCPU.ts';
import { DrawGroundPass } from './passes/drawGroundPass.ts';
import { Camera2 } from './web/camera2.ts';
import { SortPassCPU_Naive } from './passes/sortPassCPU_Naive.ts';
import { PassCtx } from './passes/passCtx.ts';

export interface ShadersTexts {
  splatPassShader: string;
  // sort
  sortBitonicShader: string;
  sortCalcDeptsShader: string;
  sortUnrollIndicesShader: string;
}

/** Web and Deno handle files differently. A bit awkward but good enough. */
export function injectShaderTexts(texts: ShadersTexts) {
  RenderSplatsGEO.SHADER_CODE = texts.splatPassShader;
  BitonicSort.SHADER_CODE = texts.sortBitonicShader;
  CalcDepthsPass.SHADER_CODE = texts.sortCalcDeptsShader;
  UnrollIndicesPass.SHADER_CODE = texts.sortUnrollIndicesShader;
}

export class Renderer {
  private readonly renderUniformBuffer: RenderUniformsBuffer;
  private readonly cameraCtrl: Camera2;
  private projectionMat: Mat4;

  // passes
  private readonly splatPass: RenderSplatsGEO;
  private readonly sortPassGPU: SortPassGPU;
  private readonly sortPassCPU: SortPassCPU;
  private readonly sortPassCPU_Naive: SortPassCPU_Naive;
  private readonly drawGroundPass: DrawGroundPass;

  constructor(
    device: GPUDevice,
    viewportSize: Dimensions,
    preferredCanvasFormat: GPUTextureFormat,
    splats: GaussianSplats
  ) {
    this.renderUniformBuffer = new RenderUniformsBuffer(device);

    this.splatPass = new RenderSplatsGEO(
      device,
      preferredCanvasFormat,
      this.renderUniformBuffer,
      splats
    );
    this.sortPassGPU = new SortPassGPU(device, splats);
    this.sortPassCPU = new SortPassCPU(splats);
    this.sortPassCPU_Naive = new SortPassCPU_Naive(splats);
    this.drawGroundPass = new DrawGroundPass(
      device,
      preferredCanvasFormat,
      this.renderUniformBuffer
    );

    this.cameraCtrl = new Camera2(CAMERA_CFG);
    this.projectionMat = createCameraProjectionMat(viewportSize);
  }

  updateCamera(deltaTime: number, input: Input): Mat4 {
    this.cameraCtrl.update(deltaTime, input);
  }

  onCanvasResize = (viewportSize: Dimensions) => {
    this.projectionMat = createCameraProjectionMat(viewportSize);
  };

  cmdRender(
    _ctx: Pick<
      PassCtx,
      'device' | 'cmdBuf' | 'splats' | 'profiler' | 'viewport'
    >,
    targetTexture: GPUTexture
  ) {
    const viewMatrix = this.cameraCtrl.viewMatrix;
    const mvpMatrix = getModelViewProjectionMatrix(
      viewMatrix,
      this.projectionMat
    );
    const ctx: PassCtx = {
      ..._ctx,
      viewMatrix,
      mvpMatrix,
      projMatrix: this.projectionMat,
      focalX: this.projectionMat[0] * _ctx.viewport.width * 0.5,
      focalY: this.projectionMat[5] * _ctx.viewport.height * 0.5,
    };

    // sort by depth
    // https://www.sctheblog.com/blog/gaussian-splatting/#sorting-and-the-index-buffer
    if (CONFIG.sortMethod === 'CPU') {
      this.sortPassCPU.cmdSortByDepth(ctx);
    } else if (CONFIG.sortMethod === 'CPU_NAIVE') {
      this.sortPassCPU_Naive.cmdSortByDepth(ctx);
    } else {
      this.sortPassGPU.cmdSortByDepth(ctx);
    }

    this.renderUniformBuffer.update(ctx);

    let splatsLoadOp: GPULoadOp = 'clear';
    if (CONFIG.drawGround) {
      this.drawGroundPass.draw(ctx, targetTexture);
      splatsLoadOp = 'load';
    }
    this.splatPass.draw(ctx, targetTexture, splatsLoadOp);
  }
}
