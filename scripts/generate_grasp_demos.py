"""
Grasp-demo ORACLE for LIBERO-Object.

Scripts privileged-state top-down grasps (approach -> descend -> close -> lift ->
carry -> release) using object/basket pose read straight from obs. Records the
RGB observations + proprio + actions of each rollout and writes ONLY successful,
reasonably-smooth demos in the exact HDF5 schema the trainer reads:

  data/demo_{k}/obs/agentview_rgb   [T,128,128,3] uint8
                 /eye_in_hand_rgb   [T,128,128,3] uint8
                 /ee_pos [T,3] /ee_ori [T,3] /gripper_states [T,2] /joint_states [T,7]
  data/demo_{k}/actions             [T,7]

The privileged pose is used ONLY to generate the trajectory, never fed to the
model -> the policy still learns RGB->action, no train/test mismatch.
"""
import os, sys, argparse, json
sys.path.insert(0, r"C:\code\LIBERO")
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import numpy as np
import h5py
from libero.libero import benchmark
from robosuite.utils.transform_utils import quat2axisangle

# task language -> object pos key stem
OBJ_ALIASES = {
    "alphabet soup": "alphabet_soup_1", "cream cheese": "cream_cheese_1",
    "salad dressing": "salad_dressing_1", "bbq sauce": "bbq_sauce_1",
    "ketchup": "ketchup_1", "tomato sauce": "tomato_sauce_1",
    "butter": "butter_1", "milk": "milk_1", "chocolate pudding": "chocolate_pudding_1",
    "orange juice": "orange_juice_1",
}


def target_key(language, obs):
    for phrase, key in OBJ_ALIASES.items():
        if phrase in language.lower() and f"{key}_pos" in obs:
            return key
    # fallback: any *_pos that isn't basket/robot, nearest to language tokens
    cands = [k[:-4] for k in obs if k.endswith("_pos") and "basket" not in k and "robot" not in k]
    for c in cands:
        if c.split("_1")[0].replace("_", " ") in language.lower():
            return c
    raise RuntimeError(f"no target object found for '{language}' in {cands}")


def script_rollout(env, obj_key, init_state, max_steps, gain, grasp_z, jiggle, rng):
    """Run one scripted grasp. Returns (success, traj dict) or (False, None)."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    rec = {k: [] for k in ["agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori",
                            "gripper_states", "joint_states", "actions"]}
    # small per-demo target perturbation -> grasp diversity
    noise = rng.uniform(-jiggle, jiggle, size=2)
    phase, hold = "REACH", 0
    APPROACH_DZ = 0.10
    z_hist = []  # for stall detection during DESCEND
    obj_z0, max_lift, dbg = None, 0.0, {}

    for step in range(max_steps):
        eef = obs["robot0_eef_pos"]
        obj_real = obs[f"{obj_key}_pos"]
        if obj_z0 is None:
            obj_z0 = float(obj_real[2])
        max_lift = max(max_lift, float(obj_real[2]) - obj_z0)
        obj = obj_real.copy(); obj[:2] += noise
        basket = obs["basket_1_pos"]
        grip = -1.0  # open by default

        if phase == "REACH":      # above the object, gripper open
            tgt = np.array([obj[0], obj[1], obj[2] + APPROACH_DZ])
            if np.linalg.norm(eef[:2] - obj[:2]) < 0.008 and abs(eef[2] - tgt[2]) < 0.03:
                phase, z_hist = "DESCEND", []
        elif phase == "DESCEND":  # lower onto grasp point (human grasps ~ at obj_z)
            tgt = np.array([obj[0], obj[1], obj[2] + grasp_z]); grip = -1.0
            z_hist.append(eef[2])            # count descend steps
            reached = (eef[2] - (obj[2] + grasp_z)) < 0.012
            if reached or len(z_hist) >= 60:  # no stall; descend up to 60 steps
                phase, hold = "CLOSE", 0
                dbg["grasp_eef_z"] = round(float(eef[2]), 4)
                dbg["min_descend_z"] = round(float(min(z_hist)), 4)
                dbg["descend_steps"] = len(z_hist)
                dbg["grasp_dxy"] = round(float(np.linalg.norm(eef[:2] - obj_real[:2])), 4)
        elif phase == "CLOSE":    # close gripper, hold still until clamped
            tgt = np.array([eef[0], eef[1], eef[2]]); grip = 1.0; hold += 1
            if hold >= 25:
                phase = "LIFT"
                dbg["grip_q"] = [round(float(x), 3) for x in obs["robot0_gripper_qpos"]]
                dbg["obj_z_atlift"] = round(float(obj_real[2]), 4)
        elif phase == "LIFT":     # raise straight up, keep closed
            tgt = np.array([eef[0], eef[1], obj[2] + 0.20]); grip = 1.0
            if eef[2] > obj[2] + 0.16:
                phase = "CARRY"
        elif phase == "CARRY":    # move over basket, keep closed
            tgt = np.array([basket[0], basket[1], basket[2] + 0.20]); grip = 1.0
            if np.linalg.norm(eef[:2] - basket[:2]) < 0.04:
                phase, hold = "RELEASE", 0
        else:                     # RELEASE
            tgt = np.array([basket[0], basket[1], basket[2] + 0.20]); grip = -1.0

        dpos = np.clip((tgt - eef) * gain, -1.0, 1.0)
        action = np.concatenate([dpos, np.zeros(3), [grip]]).astype(np.float32)

        # record BEFORE stepping (obs at decision time)
        rec["agentview_rgb"].append(obs["agentview_image"][::-1])      # LIBERO stores flipped
        rec["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"][::-1])
        rec["ee_pos"].append(obs["robot0_eef_pos"].astype(np.float32))
        rec["ee_ori"].append(quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32))
        rec["gripper_states"].append(obs["robot0_gripper_qpos"].astype(np.float32))
        rec["joint_states"].append(obs["robot0_joint_pos"].astype(np.float32))
        rec["actions"].append(action)

        obs, _, done, _ = env.step(action)
        if env.check_success():
            return True, {k: np.asarray(v) for k, v in rec.items()}, phase, step + 1, dbg
    of = obs[f"{obj_key}_pos"]; bk = obs["basket_1_pos"]
    dbg["max_lift"] = round(max_lift, 3)
    dbg["obj_basket_dxy"] = round(float(np.linalg.norm(of[:2] - bk[:2])), 3)
    return False, None, phase, max_steps, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_idx", type=int, default=1)      # 1 = cream cheese
    ap.add_argument("--n_demos", type=int, default=3)        # successful demos to collect
    ap.add_argument("--max_attempts", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--gain", type=float, default=15.0)
    ap.add_argument("--grasp_z", type=float, default=0.0)    # z offset at grasp
    ap.add_argument("--jiggle", type=float, default=0.0)     # xy target noise (diversity)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bench = benchmark.get_benchmark_dict()["libero_object"]()
    task = bench.get_task(args.task_idx)
    print("task:", task.language)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(args.seed)
    init_states = bench.get_task_init_states(args.task_idx)
    rng = np.random.RandomState(args.seed)

    obs = env.reset()
    okey = target_key(task.language, obs)
    print("target object key:", okey)

    demos, attempts, phases = [], 0, []
    while len(demos) < args.n_demos and attempts < args.max_attempts:
        ist = init_states[attempts % init_states.shape[0]]
        ok, traj, lastphase, nsteps, dbg = script_rollout(env, okey, ist, args.max_steps,
                                                           args.gain, args.grasp_z, args.jiggle, rng)
        attempts += 1
        phases.append(lastphase)
        status = "OK " if ok else "fail"
        print(f"  attempt {attempts:2d}: {status} ({lastphase}, {nsteps} st) {dbg}")
        if ok:
            demos.append(traj)
    env.close()

    print(f"\n=== {len(demos)}/{attempts} successful  | last-phase histogram: "
          f"{ {p: phases.count(p) for p in set(phases)} } ===")

    if args.out and demos:
        with h5py.File(args.out, "w") as f:
            g = f.create_group("data")
            for k, traj in enumerate(demos):
                dg = g.create_group(f"demo_{k}")
                og = dg.create_group("obs")
                og.create_dataset("agentview_rgb", data=traj["agentview_rgb"].astype(np.uint8))
                og.create_dataset("eye_in_hand_rgb", data=traj["eye_in_hand_rgb"].astype(np.uint8))
                og.create_dataset("ee_pos", data=traj["ee_pos"])
                og.create_dataset("ee_ori", data=traj["ee_ori"])
                og.create_dataset("gripper_states", data=traj["gripper_states"])
                og.create_dataset("joint_states", data=traj["joint_states"])
                dg.create_dataset("actions", data=traj["actions"])
        print("wrote", args.out)


if __name__ == "__main__":
    main()
