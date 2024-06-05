struct Uniforms {
    mvpMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    projMatrix: mat4x4<f32>,
    viewportAndFocals: vec4f,
    scaleModifier: f32,
};
@binding(0) @group(0)
var<uniform> _uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> _splatPositions: array<vec4f>;
@group(0) @binding(2)
var<storage, read> _splatRotations: array<vec4f>;
@group(0) @binding(3)
var<storage, read> _splatScales: array<vec4f>;
@group(0) @binding(4)
var<storage, read> _splatColors: array<u32>;


fn checkIsCulled(projectedPosition: vec4f) -> bool {
  let x = projectedPosition.x;
  let y = projectedPosition.y;
  let z = projectedPosition.z;
  let clipZ = projectedPosition.w;
  let clip = 1.2 * projectedPosition.w;
  return z < -clipZ || z > clipZ ||
    x < -clip || x > clip ||
    y < -clip || y > clip;
}

/**
 * https://www.sctheblog.com/blog/gaussian-splatting/#covariance-3d
 */
fn sigmaFromScaleAndRotation(scale: vec3<f32>, rot: vec4<f32>) -> mat3x3<f32> {
    let sMod = _uniforms.scaleModifier;
    let S = mat3x3<f32>(
        sMod*scale.x, 0.0, 0.0,
        0.0, sMod*scale.y, 0.0,
        0.0, 0.0, sMod*scale.z,
    );

    // quaternion to rotation matrix
    // Eq. 10, the usual stuff
    // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    let r = rot.x;
    let x = rot.y; // i
    let y = rot.z; // j
    let z = rot.w; // k
    var R = mat3x3<f32>(
        1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
        2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
        2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
    );
    R = transpose(R); // invert (it's rotation matrix after all). Left/right-handed conversion when compared to paper.

    // Equation 6 from "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    // "The covariance matrix Σ of a 3D Gaussian is analogous to describing
    // the configuration of an ellipsoid. Given a scaling matrix S and
    // rotation matrix R, we can find the corresponding Σ:
    // $\Sigma = R * S * S^T * R^T$"
    // PS. We are transposing scale matrix. O tempora, o mores!
    let Sigma = R * S * transpose(S) * transpose(R);
    return Sigma;
} 

/** 
 * https://www.sctheblog.com/blog/gaussian-splatting/#covariance-2d
 */
fn sigmaPrimCov2d(position: vec3<f32>, scale: vec3<f32>, rot: vec4<f32>) -> mat2x2<f32> {
    let sigma3x3 = sigmaFromScaleAndRotation(scale, rot);

    // t := posViewSpace
    var t = _uniforms.viewMatrix * vec4<f32>(position, 1.0);

    // Jacobian of the affine approximation of the projective transformation
    let focalX = _uniforms.viewportAndFocals[2];
    let focalY = _uniforms.viewportAndFocals[3];
    var J = mat4x4(
        focalX / t.z, 0.,           -(focalX * t.x) / (t.z * t.z), 0.,
        0.,           focalY / t.z, -(focalY * t.y) / (t.z * t.z), 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
    );
    // J = _uniforms.projMatrix; // For blog post photo
    J = transpose(J); // Left/right-handed conversion when compared to paper. Must match viewMatrix's

    // Equation (5)
    // $\sigma' = J * W * \sigma * W^T * J^T$
    // We go into the world space, apply sigma (so rotation+scale transforms),
    // and then go back
    let W = _uniforms.viewMatrix;
    let sigma4x4 = mat4x4(
        // covariance matrices are symmetric, so I don't have to care if row-col idx order is ok
        sigma3x3[0][0], sigma3x3[0][1], sigma3x3[0][2], 0.,
        sigma3x3[1][0], sigma3x3[1][1], sigma3x3[1][2], 0.,
        sigma3x3[2][0], sigma3x3[2][1], sigma3x3[2][2], 0.,
        0., 0., 0., 0.,
    );
    var sigma2d = J * W * sigma4x4 * transpose(W) * transpose(J);

    // Copied from reference impl.
    // https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L108
    // "Apply low-pass filter: every Gaussian should be at least one pixel wide/high."
    // It probably also guarantees sigma2d is invertible?
    sigma2d[0][0] += 0.3;
    sigma2d[1][1] += 0.3;

    // Discard 3rd and 4th row and column.
    // From "EWA Splatting", below equation (23) - https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf :
    // "The 2x2 variance matrix V^ is easily obtained from the 3D matrix V
    // by skipping the third row and column"
    // New matrix is a bit wasteful, but matches math nicer
    let sigmaPrim = mat2x2(
      sigma2d[0][0], sigma2d[0][1],
      sigma2d[1][0], sigma2d[1][1],
    );
    return sigmaPrim;
}

/**
 * https://www.sctheblog.com/blog/gaussian-splatting/#calculating-eigenvalues
 * 
 * I remember in high school they said:
 * "When in the real world you will make use of quadratic equation?"
 */
fn calcEigenvalues(a: f32, b: f32, c: f32) -> vec2f {
  let delta = sqrt(a*a - 2*a*c + 4*b*b + c*c);
  let lambda1 = 0.5 * (a+c + delta);
  let lambda2 = 0.5 * (a+c - delta);
  return vec2f(lambda1, lambda2);
}

const BILLBOARD_VERTICES = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
);

struct Gaussian {
  vertexInSplatIdx: u32,
  splatIdx: u32,
  worldPos: vec4f,
  scale: vec3f,
  rot: vec4f,
  color: vec4f,
  quadOffset: vec2f,
};

fn readGaussian(in_vertex_index: u32) -> Gaussian {
  var result: Gaussian;
  result.vertexInSplatIdx = in_vertex_index % 6u;
  result.splatIdx = in_vertex_index / 6u;
  result.worldPos = _splatPositions[result.splatIdx];
  result.scale = _splatScales[result.splatIdx].xyz;
  result.rot = _splatRotations[result.splatIdx];
  let splatColor = _splatColors[result.splatIdx];
  result.color = vec4<f32>(
    f32((splatColor >> 24) & 0xff) / 255.0,
    f32((splatColor >> 16) & 0xff) / 255.0,
    f32((splatColor >> 8 ) & 0xff) / 255.0,
    f32((splatColor      ) & 0xff) / 255.0,
  );
  result.quadOffset = BILLBOARD_VERTICES[result.vertexInSplatIdx];
  return result;
}


///////////////////////////
/// Eigenvectors version
/// https://www.sctheblog.com/blog/gaussian-splatting/#method-2-calculate-eigenvectors

struct VertexOutputEigenvectors {
  @builtin(position) position: vec4<f32>,
  @location(0) splatColor: vec4<f32>,
  @location(1) quadOffset: vec2<f32>,
};

@vertex
fn vs_mainEigenvectors(
  @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutputEigenvectors {
  var result: VertexOutputEigenvectors;
  let g = readGaussian(in_vertex_index);
  let viewportWidth = _uniforms.viewportAndFocals[0];
  let viewportHeight =_uniforms.viewportAndFocals[1];
  let viewport = vec2(viewportWidth, viewportHeight);

  // projection
  var projectedPosition = _uniforms.mvpMatrix * g.worldPos;
  if (checkIsCulled(projectedPosition)) {
    result.splatColor.w = 0.0;
    return result;
  }
  projectedPosition /= projectedPosition.w;

  // cov2d = [a  b]  <-- ASCII 2D matrix
  //         [b  c]
  let cov2d = sigmaPrimCov2d(g.worldPos.xyz, g.scale.xyz, g.rot);
  let a = cov2d[0][0];
  let b = cov2d[0][1];
  let c = cov2d[1][1];

  let eigenvalues = calcEigenvalues(a, b, c);
  let lambda1 = eigenvalues.x;
  let lambda2 = eigenvalues.y;
  
  // Use eigenvectors to calc major/minor axis
  // https://www.sctheblog.com/blog/gaussian-splatting/#vertex-shader-eigenvectors
  let diagonalVector = normalize(vec2(1, (-a+b+lambda1) / (b-c+lambda1) ));
  let diagonalVectorOther = vec2(diagonalVector.y, -diagonalVector.x);
  let majorAxis = min(3 * sqrt(lambda1), 1024) * diagonalVector;
  let minorAxis = min(3 * sqrt(lambda2), 1024) * diagonalVectorOther;
  var projectedPosition2D = projectedPosition.xy; // WGSL..
  projectedPosition2D += g.quadOffset.x * majorAxis / viewport 
                       + g.quadOffset.y * minorAxis / viewport;
  result.position = vec4<f32>(projectedPosition2D, 0.0, 1.0);
  result.quadOffset = g.quadOffset;

  // finish
  result.splatColor = g.color;
  return result;
}

@fragment
fn fs_mainEigenvectors(fragIn: VertexOutputEigenvectors) -> @location(0) vec4<f32> {
  // "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
  // $\alpha_i$ in eq.3 in 
  // $\alpha_j$ is done through blending mode during rendering
  let opacity = fragIn.splatColor.w;

  // https://www.sctheblog.com/blog/gaussian-splatting/#fragment-shader
  // this is for circle, but our vertices projected it into an ellipsis
  let r = dot(fragIn.quadOffset, fragIn.quadOffset);
  if (r > 1.0){ discard; }
  // "The exponential function in (9) can now be written as $e^{−0.5*r}$",
  // where (9) is a definition for:
  // "We define an elliptical Gaussian Gv(x − p) centered at a point p with a variance matrix V as:"
  let Gv = exp(-0.5 * r);
  let a = Gv * opacity;
  return vec4(a * fragIn.splatColor.rgb, a);
}



///////////////////////////
/// Square billboard version
/// https://www.sctheblog.com/blog/gaussian-splatting/#method-1-project-gaussian-to-a-square

struct VertexOutputSquare {
  @builtin(position) position: vec4<f32>,
  @location(0) splatColor: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) conic: vec4<f32>,
};

@vertex
fn vs_mainSquare(
  @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutputSquare {
  var result: VertexOutputSquare;
  let g = readGaussian(in_vertex_index);
  let viewportWidth = _uniforms.viewportAndFocals[0];
  let viewportHeight =_uniforms.viewportAndFocals[1];

  // projection
  var projectedPosition = _uniforms.mvpMatrix * g.worldPos;
  if (checkIsCulled(projectedPosition)) {
    result.splatColor.w = 0.0;
    return result;
  }
  projectedPosition /= projectedPosition.w;

  // cov2d = [a  b]  <-- ASCII 2D matrix
  //         [b  c]
  let cov2d = sigmaPrimCov2d(g.worldPos.xyz, g.scale.xyz, g.rot);
  let a = cov2d[0][0];
  let b = cov2d[0][1];
  let c = cov2d[1][1];

  let eigenvalues = calcEigenvalues(a, b, c);
  let lambda1 = eigenvalues.x;
  let lambda2 = eigenvalues.y;

  // Rendering as square (based on tiled) (draft)
  // This is slighly adopted as I'm not using the tiles
  // Uses max(λ1, λ2) as a radius to 'spread' the billboard vertices.
  // https://www.sctheblog.com/blog/gaussian-splatting/#calculate-tiles
  let radiusPx = ceil(3. * sqrt(max(lambda1, lambda2)));
  let radiusNdc = vec2<f32>(
    // This magic 2.0 is the reason why this mode is experimental (although it works).
    // Not sure which value is incorrect, the math seems ok. Maybe focal?
    2.0 * radiusPx / viewportWidth,
    2.0 * radiusPx / viewportHeight
  );
  result.position = vec4<f32>(projectedPosition.xy + radiusNdc * g.quadOffset, projectedPosition.zw);
  result.uv = radiusPx * g.quadOffset; // in pixels!

  // conic (inverse of cov2d)
  let det = a * c - b * b; // https://www.w3.org/TR/WGSL/#determinant-builtin
  let conic = vec3<f32>(c / det, -b / det, a / det);
  result.conic = vec4<f32>(conic, 0.0);

  // finish
  result.splatColor = g.color;
  return result;
}

@fragment
fn fs_mainSquare(fragIn: VertexOutputSquare) -> @location(0) vec4<f32> {
  let opacity = fragIn.splatColor.w;

  // Rendering as square (based on tiled) (draft)
  // https://www.sctheblog.com/blog/gaussian-splatting/#calculate-tiles
  let d = -fragIn.uv; // quadOffset in pixels
  let conic = fragIn.conic.xyz; // [0][0], [0][1] and [1][0], [1],[1]
  let power = -0.5 * (
    conic.x * d.x * d.x +
    conic.z * d.y * d.y
    ) - conic.y * d.x * d.y;

  if (power > 0.0) { discard; }

  // Eq. (2) from 3D Gaussian splatting paper.
  // Obtain alpha by multiplying with Gaussian opacity
  // and its exponential falloff from mean.
  let alpha = min(0.99, opacity * exp(power));
  return vec4<f32>(fragIn.splatColor.xyz * alpha, alpha);
}
