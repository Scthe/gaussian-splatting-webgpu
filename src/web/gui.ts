import * as dat from 'dat.gui';
import { CONFIG } from '../constants.ts';
import { GpuProfiler, GpuProfilerResult } from '../gpuProfiler.ts';

// https://github.com/Scthe/WebFX/blob/master/src/UISystem.ts#L13

export function initializeGUI(profiler: GpuProfiler) {
  const gui = new dat.GUI();

  const dummyObject = {
    openGithub: () => {
      window.location.href = CONFIG.githubRepoLink;
    },
    profile: () => {
      profiler.profileNextFrame(true);
    },
  };

  // github
  gui.add(dummyObject, 'openGithub').name('GITHUB');

  // bg
  addColorController(CONFIG, 'clearColor', 'Bg color');

  // sorting method
  const sortModeDummy = createDummy(CONFIG, 'sortMethod', [
    { label: 'GPU', value: 'GPU' },
    { label: 'CPU Counting sort', value: 'CPU' },
    { label: 'CPU .sort', value: 'CPU_NAIVE' },
  ]);
  gui.add(sortModeDummy, 'sortMethod', sortModeDummy.values).name('Sorting');

  // render method
  const renderModeDummy = createDummy(CONFIG, 'renderMethod', [
    { label: 'Eigenvectors', value: 'EIGENVECTORS' },
    { label: 'Square billboard', value: 'SQUARE_BILLBOARD' },
  ]);
  gui
    .add(renderModeDummy, 'renderMethod', renderModeDummy.values)
    .name('Render');

  // splat scale
  gui.add(CONFIG, 'scaleModifier', 0, 2.0).name('Splat scale');

  // ground
  gui.add(CONFIG, 'drawGround').name('Draw ground');

  // profiler
  gui.add(dummyObject, 'profile').name('Profile');

  //////////////
  /// utils
  function addColorController<T extends object>(
    obj: T,
    prop: keyof T,
    name: string
  ) {
    const dummy = {
      value: [] as number[],
    };

    Object.defineProperty(dummy, 'value', {
      enumerable: true,
      get: () => {
        // deno-lint-ignore no-explicit-any
        const v = obj[prop] as any;
        return [v[0] * 255, v[1] * 255, v[2] * 255];
      },
      set: (v: number[]) => {
        // deno-lint-ignore no-explicit-any
        const a = obj[prop] as any as number[];
        a[0] = v[0] / 255;
        a[1] = v[1] / 255;
        a[2] = v[2] / 255;
      },
    });

    gui.addColor(dummy, 'value').name(name);
  }
}

interface UiOpts<T> {
  label: string;
  value: T;
}

const createDummy = <V extends Object, K extends keyof V>(
  obj: V,
  key: K,
  opts: UiOpts<V[K]>[]
) => {
  const dummy = {
    values: opts.map((o) => o.label),
  };

  Object.defineProperty(dummy, key, {
    enumerable: true,
    get: () => {
      const v = obj[key];
      const opt = opts.find((e) => e.value === v) || opts[0];
      return opt.label;
    },
    set: (selectedLabel: string) => {
      const opt = opts.find((e) => e.label === selectedLabel) || opts[0];
      obj[key] = opt.value;
    },
  });

  return dummy;
};

export function onGpuProfilerResult(result: GpuProfilerResult) {
  console.log('Profiler:', result);
  const parentEl = document.getElementById('profiler-results');
  parentEl.innerHTML = '';
  parentEl.parentNode.style.display = 'block';

  const mergeByName: Record<string, number> = {};
  const names = new Set<string>();
  result.forEach(([name, timeMs]) => {
    const t = mergeByName[name] || 0;
    mergeByName[name] = t + timeMs;
    names.add(name);
  });

  names.forEach((name) => {
    const timeMs = mergeByName[name];
    const li = document.createElement('li');
    li.innerHTML = `${name}: ${timeMs.toFixed(2)}ms`;
    parentEl.appendChild(li);
  });
}
