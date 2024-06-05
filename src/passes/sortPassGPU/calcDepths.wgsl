@group(0) @binding(0)
var<storage, read> _splatPositions: array<vec4f>;

@group(0) @binding(1)
var<storage, read_write> _distancesBuffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> _indicesBuffer: array<u32>;

@group(0) @binding(3)
var<uniform> _mvpMatrix: mat4x4<f32>;

@compute
@workgroup_size(1) // TODO?
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let splatCount = u32(__SPLAT_COUNT__);
    let itemsPerThread = u32(__ITEMS_PER_THREAD__);
    let startIdx = global_id.x * itemsPerThread;
    let endIdx = (global_id.x + 1u) * itemsPerThread;

    for (var i = startIdx; i < endIdx; i++ ) {
         if (i >= splatCount) {
            _distancesBuffer[i] = 999999.9f;
        } else {
            let pos = _splatPositions[i];
            let projPos = _mvpMatrix * pos;
            _distancesBuffer[i] = projPos.z;
            _indicesBuffer[i] = i;
        }
    }
}

