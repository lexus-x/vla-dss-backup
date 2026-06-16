"""
Fast diagnostic: create one LIBERO env, time the render, and dump the obs keys
and mujoco body names. Tells us (a) is GL on the GPU or a slow software path,
(b) what object/target positions are available for a z-error metric.

Run with MUJOCO_GL set (egl|glfw|osmesa) to compare backends.
"""
import os, sys, time
sys.path.insert(0, os.environ.get("LIBERO_SRC", "C:/code/LIBERO"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_object"]()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"task: {task.language}", flush=True)

t0 = time.time()
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
print(f"env create: {time.time()-t0:.1f}s", flush=True)
env.seed(0)
init_states = __import__("torch").load(
    os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file))
env.reset()
obs = env.set_init_state(init_states[0])

t0 = time.time()
N = 20
for _ in range(N):
    obs, _, _, _ = env.step(np.zeros(7))
dt = (time.time() - t0) / N
print(f"per-step (with render): {dt*1000:.0f} ms  ->  ~{dt*400:.1f}s per 400-step rollout", flush=True)

print("\n=== obs keys ===", flush=True)
for k in sorted(obs.keys()):
    v = obs[k]
    shp = getattr(v, "shape", None)
    print(f"  {k:35s} {shp if shp is not None else type(v).__name__}", flush=True)

print("\n=== mujoco body names (non-robot) ===", flush=True)
sim = env.env.sim
for i in range(sim.model.nbody):
    name = sim.model.body_id2name(i)
    if name and not name.startswith(("robot", "gripper", "world", "table")):
        xp = sim.data.body_xpos[i]
        print(f"  {name:35s} xpos=({xp[0]:.3f},{xp[1]:.3f},{xp[2]:.3f})", flush=True)

env.close()
print("PROBE DONE", flush=True)
