struct Uniforms {
    j: u32,
    k: u32,
};

@group(0) @binding(0)
var<storage, read_write> _indicesBuffer: array<u32>;

@group(0) @binding(1)
var<storage, read_write> _distancesBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;


// https://en.wikipedia.org/wiki/Bitonic_sorter
@compute
@workgroup_size(__WORKGROUP_SIZE__) // TODO?
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>, // [0,1,2,3, ..., WORKGROUP.xyz]
) {
    let itemsPerThread = u32(__ITEMS_PER_THREAD__);
    let startIdx = global_id.x * itemsPerThread;
    let endIdx = (global_id.x + 1) * itemsPerThread;
    let j = uniforms.j;
    let k = uniforms.k;

    // Single dispatch impossible, lack of global barriers in WGSL
    // https://stackoverflow.com/questions/72035548/what-does-storagebarrier-in-webgpu-actually-do
    // let elementCount = 524288u;
    // for (var k = 2u; k <= elementCount; k <<= 1u) {
    // for (var j = k >> 1u; j > 0u; j >>= 1u) {

    for (var i = startIdx; i < endIdx; i++ ) {
        let i_XOR_j = i ^ j; // WIKIPEDIA: this is 'l' on wikipedia
        if (i_XOR_j <= i) {
            continue;
        }
        let swap0 = (i & k) == 0 && _distancesBuffer[i] > _distancesBuffer[i_XOR_j];
        let swap1 = (i & k) != 0 && _distancesBuffer[i] < _distancesBuffer[i_XOR_j];
        if (swap0 || swap1) {
            // WIKIPEDIA: swap the elements arr[i] and arr[i_XOR_j]
            let tmp0 = _distancesBuffer[i];
            _distancesBuffer[i] = _distancesBuffer[i_XOR_j];
            _distancesBuffer[i_XOR_j] = tmp0;
            
            let tmp1 = _indicesBuffer[i];
            _indicesBuffer[i] = _indicesBuffer[i_XOR_j];
            _indicesBuffer[i_XOR_j] = tmp1;
        }
    }

    // workgroupBarrier(); // barriers snippet
    // storageBarrier();
    // }}
}
