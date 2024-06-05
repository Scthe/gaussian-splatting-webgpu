import { CONFIG } from '../constants.ts';
import { PassCtx } from './passCtx.ts';
import { RenderUniformsBuffer } from './renderUniformsBuffer.ts';

const CODE = `
struct Uniforms {
  mvpMatrix: mat4x4<f32>,
  viewMatrix: mat4x4<f32>,
  scaleModifier: f32,
};
@binding(0) @group(0)
var<uniform> _uniforms: Uniforms;

const H:f32 = 0.0;

@vertex
fn ground_main_vs(
  @builtin(vertex_index) in_vertex_index : u32
) -> @builtin(position) vec4f {
  let squareScale = 1.0;
  var offsetX = f32(i32(in_vertex_index / 2u ) * 2 - 1); // [-1, -1, 0, 0]
  var offsetY = f32(i32(in_vertex_index & 1u) * 2 - 1); // [-1, 1, -1, 1, ...]
  offsetX = offsetX * squareScale;
  offsetY = offsetY * squareScale;

  var pos = vec4<f32>(offsetX, H, offsetY, 1.);
  var projectedPosition = _uniforms.mvpMatrix * pos;
  projectedPosition /= projectedPosition.w;
  return projectedPosition;
}

@fragment
fn ground_main_fs() -> @location(0) vec4f {
  let c = 0.9;
  return vec4(c, c, c, 1.0);
}
`;

export class DrawGroundPass {
  public static NAME: string = DrawGroundPass.name;

  private readonly renderPipeline: GPURenderPipeline;
  public readonly uniformsBindings: GPUBindGroup;

  constructor(
    device: GPUDevice,
    outTextureFormat: GPUTextureFormat,
    uniforms: RenderUniformsBuffer
  ) {
    this.renderPipeline = DrawGroundPass.createRenderPipeline(
      device,
      outTextureFormat
    );
    this.uniformsBindings = DrawGroundPass.assignResourcesToBindings(
      device,
      this.renderPipeline.getBindGroupLayout(0),
      uniforms
    );
  }

  private static createRenderPipeline(
    device: GPUDevice,
    outTextureFormat: GPUTextureFormat
  ) {
    const shaderModule = device.createShaderModule({ code: CODE });

    const renderPipeline = device.createRenderPipeline({
      label: 'splats-render',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'ground_main_vs',
        buffers: [],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'ground_main_fs',
        targets: [{ format: outTextureFormat }],
      },
      primitive: {
        cullMode: 'none',
        topology: 'triangle-strip',
        stripIndexFormat: undefined,
      },
    });

    return renderPipeline;
  }

  private static assignResourcesToBindings(
    device: GPUDevice,
    uniformsLayout: GPUBindGroupLayout,
    uniforms: RenderUniformsBuffer
  ) {
    return device.createBindGroup({
      layout: uniformsLayout,
      entries: [uniforms.createBindingDesc(0)],
    });
  }

  draw(ctx: PassCtx, targetTexture: GPUTexture) {
    const { cmdBuf, profiler } = ctx;

    // https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/beginRenderPass
    const renderPass = cmdBuf.beginRenderPass({
      label: DrawGroundPass.NAME,
      colorAttachments: [
        {
          view: targetTexture.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [
            CONFIG.clearColor[0],
            CONFIG.clearColor[1],
            CONFIG.clearColor[2],
            0, // important for blending!
          ],
        },
      ],
      timestampWrites: profiler?.createScopeGpu(DrawGroundPass.NAME),
    });

    // set render pass data
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.uniformsBindings);
    renderPass.draw(4, 1, 0, 0);
    renderPass.end();
  }
}
