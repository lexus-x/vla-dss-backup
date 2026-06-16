"""Verify target_obj_key resolves to a REAL object key present in each task's obs.
If it returns None or a wrong/static key, the z-error/lift metrics are invalid."""
import os, sys
sys.path.insert(0, os.environ.get("LIBERO_SRC", "C:/code/LIBERO"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scripts.eval_sim import target_obj_key
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_object"]()
for ti in range(5):
    task = bench.get_task(ti)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0)
    init = __import__("torch").load(os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file))
    env.reset(); obs = env.set_init_state(init[0])
    for _ in range(5): obs,_,_,_ = env.step(np.zeros(7))
    pos_keys = [k for k in obs if k.endswith('_pos') and 'to_robot' not in k and 'eef' not in k]
    resolved = target_obj_key(task.language, obs)
    ok = "OK" if (resolved and resolved in obs) else ">>> BAD <<<"
    print(f"[{ti}] {task.language}")
    print(f"     resolved obj_key = {resolved}   [{ok}]")
    print(f"     pos keys in scene: {pos_keys}")
    env.close()
print("DONE")
