@group(0) @binding(0)
var<storage, read> _indicesBuffer: array<u32>;

@group(0) @binding(1)
var<storage, read_write> _unrolledIndices: array<u32>;

@compute
@workgroup_size(1) // TODO?
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let itemsPerThread = u32(__ITEMS_PER_THREAD__);
    let startIdx = global_id.x * itemsPerThread;
    let endIdx = (global_id.x + 1u) * itemsPerThread;
    let verticesPerSplat = u32(__VERTICES_PER_SPLAT__);

    for (var i = startIdx; i < endIdx; i++) {
        let idx = _indicesBuffer[i];
        for (var j = 0u; j < verticesPerSplat; j++) {
            _unrolledIndices[i * verticesPerSplat + j] = idx * verticesPerSplat + j;
        }
    }
}