"""
Octo-Small zero-shot / finetuned eval on LIBERO-Object, matching our protocol
(OffScreenRenderEnv, execute-horizon open-loop, per-task success). Runs in WSL.

Zero-shot caveats: Octo's action is unnormalized with a chosen pretraining dataset's
stats (--unnorm_key) and mapped to LIBERO's OSC_POSE delta convention. Mismatch is
expected -> zero-shot likely low. Fine-tuned eval uses the finetune ckpt's own stats.
"""
import os, sys, time, argparse, json
sys.path.insert(0, "/mnt/c/code/LIBERO")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import jax
import cv2
from collections import deque
from octo.model.octo_model import OctoModel


def resize(img, s):
    return cv2.resize(np.ascontiguousarray(img), (s, s), interpolation=cv2.INTER_AREA)


def perturb_img(img, kind, sev):
    """IDENTICAL to eval_sim.py perturb_img. img: HxWx3 uint8 RGB."""
    if kind == 'none' or sev <= 0:
        return img
    im = img.astype(np.float32)
    if kind == 'noise':
        std = [0, 5, 10, 16, 24, 34][min(sev, 5)]
        im = im + np.random.normal(0.0, std, im.shape)
    elif kind == 'blur':
        sigma = [0, 0.6, 1.1, 1.8, 2.7, 4.0][min(sev, 5)]
        kk = max(1, int(2 * round(sigma) + 1)) | 1
        im = cv2.GaussianBlur(im, (kk, kk), sigma)
    elif kind == 'brightness':
        im = im + [0, 18, 36, 58, 84, 115][min(sev, 5)]
    return np.clip(im, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned_path", default="")          # empty -> zero-shot pretrained
    ap.add_argument("--step", type=int, default=-1)          # checkpoint step for finetuned dir
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--n_tasks", type=int, default=2)
    ap.add_argument("--n_rollouts", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=220)
    ap.add_argument("--exec_horizon", type=int, default=4)
    ap.add_argument("--unnorm_key", default="bridge_dataset")  # zero-shot action stats
    ap.add_argument("--perturb", default="none")               # robustness: none|noise|blur|brightness
    ap.add_argument("--severity", type=int, default=0)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    np.random.seed(0)   # reproducible perturbation noise

    print("devices:", jax.devices())
    path = args.finetuned_path or "hf://rail-berkeley/octo-small-1.5"
    print("loading", path, "step", args.step)
    if args.finetuned_path and args.step >= 0:
        model = OctoModel.load_pretrained(path, step=args.step)
    else:
        model = OctoModel.load_pretrained(path)
    # pick action unnormalization stats
    ds = model.dataset_statistics
    if "action" in ds:                       # finetuned model: single dataset
        act_stats = ds["action"]
    else:                                    # pretrained: dict of datasets
        print("available unnorm datasets:", list(ds.keys())[:10])
        key = args.unnorm_key if args.unnorm_key in ds else list(ds.keys())[0]
        print("using unnorm_key:", key)
        act_stats = ds[key]["action"]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import torch
    bench = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = bench.get_num_tasks() if args.n_tasks < 0 else min(args.n_tasks, bench.get_num_tasks())
    bddl_root = get_libero_path("bddl_files")

    rng = jax.random.PRNGKey(0)
    jlog = open(args.log, "a") if args.log else None
    results = {}
    for ti in range(n_tasks):
        task = bench.get_task(ti)
        bddl = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        init_states = bench.get_task_init_states(ti)
        octo_task = model.create_tasks(texts=[task.language])
        succ = 0
        for r in range(args.n_rollouts):
            env.reset()
            obs = env.set_init_state(init_states[r % init_states.shape[0]])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))
            # 2-frame history of primary (256) + wrist (128)
            prim = deque(maxlen=2); wrist = deque(maxlen=2)
            def push(o):
                # OpenVLA/RLDS convention: rotate 180 deg (vertical+horizontal flip)
                pa = perturb_img(o["agentview_image"][::-1, ::-1], args.perturb, args.severity)
                pw = perturb_img(o["robot0_eye_in_hand_image"][::-1, ::-1], args.perturb, args.severity)
                prim.append(resize(pa, 256))
                wrist.append(resize(pw, 128))
            push(obs); push(obs)   # duplicate first frame to fill window
            done = False; steps = 0
            exec_actions = []   # executed continuous 6D commands -> jerk (same as eval_sim.py)
            while steps < args.max_steps and not done:
                octo_obs = {
                    "image_primary": np.stack(prim)[None],          # [1,2,256,256,3]
                    "image_wrist": np.stack(wrist)[None],           # [1,2,128,128,3]
                    "timestep_pad_mask": np.ones((1, 2), dtype=bool),
                }
                rng, k = jax.random.split(rng)
                act = model.sample_actions(octo_obs, octo_task,
                                           unnormalization_statistics=act_stats, rng=k)
                act = np.asarray(act[0])                            # [pred_horizon, 7]
                for h in range(min(args.exec_horizon, act.shape[0])):
                    # Octo outputs the unnormalized LIBERO action directly (7-dim); feed as-is
                    a = np.clip(act[h], -1, 1)
                    exec_actions.append(np.asarray(a[:6], dtype=float).copy())
                    obs, _, _, _ = env.step(a)
                    steps += 1
                    push(obs)
                    if env.check_success():
                        done = True; break
            ok = bool(env.check_success()); succ += int(ok)
            # jerk = mean |2nd difference| of executed continuous trajectory (identical to eval_sim.py)
            ea = np.asarray(exec_actions)
            jerk_r = float(np.abs(np.diff(ea, n=2, axis=0)).mean()) if ea.shape[0] >= 3 else float('nan')
            print(f"  task{ti} r{r}: {'OK' if ok else 'fail'} ({steps} steps, jerk {jerk_r:.4f})")
            if jlog:
                jlog.write(json.dumps({"task_idx": ti, "task": task.language, "rollout": r,
                                       "perturb": args.perturb, "severity": args.severity,
                                       "success": int(ok), "steps": steps, "jerk": jerk_r}) + "\n"); jlog.flush()
        env.close()
        rate = 100.0 * succ / args.n_rollouts
        results[task.language] = rate
        print(f"[{ti+1}/{n_tasks}] {rate:.1f}% ({succ}/{args.n_rollouts})  {task.language}")
    avg = float(np.mean(list(results.values()))) if results else 0.0
    print(f"=== Octo {'finetuned' if args.finetuned_path else 'ZERO-SHOT'} | {args.suite} | "
          f"success {avg:.1f}% ({n_tasks}x{args.n_rollouts}) ===")
    if jlog: jlog.close()


if __name__ == "__main__":
    main()
