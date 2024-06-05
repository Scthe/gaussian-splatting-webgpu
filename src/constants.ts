const camDist = 2.5;

export const CAMERA_CFG = {
  // pos + rot (magic values that work for this model, do not ask)
  position: [camDist, 0, camDist],
  target: [-1.5, 3.0, 0],
  // projection
  fovDgr: 60,
  near: 0.01,
  far: 100,
};

export const BYTES_U8 = 1;
export const BYTES_F32 = 4;
export const BYTES_U32 = 4;
export const BYTES_U64 = 8;
export const BYTES_VEC4 = BYTES_F32 * 4;
export const BYTES_MAT4 = BYTES_F32 * 16;

type SortMethod = 'GPU' | 'CPU' | 'CPU_NAIVE';
type RenderMethod = 'EIGENVECTORS' | 'SQUARE_BILLBOARD';

export const CONFIG = {
  githubRepoLink: 'https://github.com/Scthe/gaussian-splatting-webgpu',
  sortMethod: 'GPU' as SortMethod,
  renderMethod: 'Eigenvectors' as RenderMethod,
  scaleModifier: 1.0,
  clearColor: [0, 0, 0],
  rotationSpeed: 1,
  movementSpeed: 2,
  drawGround: false,
};
