import { GaussianSplats, VERTICES_PER_SPLAT } from '../gaussianSplats.ts';
import { CONFIG } from '../constants.ts';
import { assertHasInjectedShader } from '../utils.ts';
import { PassCtx } from './passCtx.ts';
import { RenderUniformsBuffer } from './renderUniformsBuffer.ts';

const VERTEX_ATTRIBUTES: GPUVertexBufferLayout[] = [];

const UNIFORMS_DESC: GPUBindGroupLayoutDescriptor = {
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        minBindingSize: RenderUniformsBuffer.BUFFER_SIZE,
      },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 3,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 4,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'read-only-storage' },
    },
  ],
};

export class RenderSplatsGEO {
  public static SHADER_CODE: string = '';
  public static NAME: string = RenderSplatsGEO.name;

  private readonly uniformsLayout: GPUBindGroupLayout;
  private readonly renderPipelineEigenvectors: GPURenderPipeline;
  private readonly renderPipelineSquare: GPURenderPipeline;
  private readonly uniformsBindings: GPUBindGroup;

  constructor(
    device: GPUDevice,
    outTextureFormat: GPUTextureFormat,
    uniforms: RenderUniformsBuffer,
    splats: GaussianSplats
  ) {
    assertHasInjectedShader(RenderSplatsGEO);
    this.uniformsLayout = device.createBindGroupLayout(UNIFORMS_DESC);

    const shaderModule = device.createShaderModule({
      code: RenderSplatsGEO.SHADER_CODE,
    });

    // https://developer.mozilla.org/en-US/docs/Web/API/GPUPipelineLayout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.uniformsLayout],
    });
    this.renderPipelineEigenvectors = RenderSplatsGEO.createRenderPipeline(
      device,
      pipelineLayout,
      shaderModule,
      outTextureFormat,
      'vs_mainEigenvectors',
      'fs_mainEigenvectors'
    );
    this.renderPipelineSquare = RenderSplatsGEO.createRenderPipeline(
      device,
      pipelineLayout,
      shaderModule,
      outTextureFormat,
      'vs_mainSquare',
      'fs_mainSquare'
    );

    this.uniformsBindings = RenderSplatsGEO.assignResourcesToBindings(
      device,
      this.uniformsLayout,
      uniforms,
      splats
    );
  }

  private static createRenderPipeline(
    device: GPUDevice,
    pipelineLayout: GPUPipelineLayout,
    shaderModule: GPUShaderModule,
    outTextureFormat: GPUTextureFormat,
    vertexShaderEntryPoint: string,
    fragmentShaderEntryPoint: string
  ) {
    return device.createRenderPipeline({
      label: `splats-render-${vertexShaderEntryPoint}`,
      layout: pipelineLayout,
      // layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: vertexShaderEntryPoint,
        buffers: VERTEX_ATTRIBUTES,
      },
      fragment: {
        module: shaderModule,
        entryPoint: fragmentShaderEntryPoint,
        targets: [
          {
            format: outTextureFormat,
            blend: {
              // Previous fragment at the pixel had written color with alpha A.
              // The next fragment behind it (sorting!) will have weight (1-A).
              //
              // See eq.3 in "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
              // the $\prod_{j=1}^{i-1} 1 - \alpha_j$ part.
              // It's exactly same in NERFs, literary everything uses this equation.
              //
              // https://www.sctheblog.com/blog/gaussian-splatting/#blending
              // https://www.sctheblog.com/blog/gaussian-splatting/#blending-settings
              color: {
                srcFactor: 'one-minus-dst-alpha',
                dstFactor: 'one',
                operation: 'add',
              },
              alpha: {
                srcFactor: 'one-minus-dst-alpha',
                dstFactor: 'one',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: {
        cullMode: 'none',
        topology: 'triangle-list',
        stripIndexFormat: undefined,
      },
    });
  }

  private static assignResourcesToBindings(
    device: GPUDevice,
    uniformsLayout: GPUBindGroupLayout,
    uniforms: RenderUniformsBuffer,
    splats: GaussianSplats
  ) {
    return device.createBindGroup({
      layout: uniformsLayout,
      entries: [
        uniforms.createBindingDesc(0),
        {
          binding: 1,
          resource: { buffer: splats.positionsBuffer },
        },
        {
          binding: 2,
          resource: { buffer: splats.rotationsBuffer },
        },
        {
          binding: 3,
          resource: { buffer: splats.scalesBuffer },
        },
        {
          binding: 4,
          resource: { buffer: splats.colorsBuffer },
        },
      ],
    });
  }

  draw(ctx: PassCtx, targetTexture: GPUTexture, loadOp: GPULoadOp) {
    const { cmdBuf, splats, profiler } = ctx;

    // https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/beginRenderPass
    const renderPass = cmdBuf.beginRenderPass({
      label: RenderSplatsGEO.NAME,
      colorAttachments: [
        {
          view: targetTexture.createView(),
          loadOp,
          storeOp: 'store',
          clearValue: [
            CONFIG.clearColor[0],
            CONFIG.clearColor[1],
            CONFIG.clearColor[2],
            0, // important for blending!
          ],
        },
      ],
      timestampWrites: profiler?.createScopeGpu(RenderSplatsGEO.NAME),
    });

    // set render pass data
    const pipeline =
      CONFIG.renderMethod === 'SQUARE_BILLBOARD'
        ? this.renderPipelineSquare
        : this.renderPipelineEigenvectors;
    // renderPass.pushDebugGroup('Prepare data for draw.');
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, this.uniformsBindings);
    renderPass.setIndexBuffer(splats.indicesBuffer, 'uint32');
    // renderPass.popDebugGroup();
    // renderPass.insertDebugMarker('Draw');

    // draw
    const vertexCount = splats.count * VERTICES_PER_SPLAT;
    renderPass.drawIndexed(vertexCount, 1, 0, 0, 0);

    // fin
    renderPass.end();
  }
}
