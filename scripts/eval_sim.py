"""
Real LIBERO simulator rollout eval (success rate).

Uses the correct LIBERO API (OffScreenRenderEnv + init_states) and maps the
live-env observation keys to exactly what the model was trained on:
  agentview_image            -> agentview view
  robot0_eye_in_hand_image   -> wrist view
  robot0_eef_pos (3)         -> ee_pos
  robot0_eef_quat (4)        -> ee_ori via quat2axisangle (3)   [matches HDF5]
  robot0_gripper_qpos (2)    -> gripper
  robot0_joint_pos (7)       -> joints
Proprio assembled as [ee_pos, ee_ori(axisangle), gripper, joints] = 15D.

Run:
  PYTHONPATH=C:/Users/DELL/.venv/LIBERO MUJOCO_GL=glfw \
  python scripts/eval_sim.py --n_rollouts 10
"""
import os, sys, time, argparse
# Ensure LIBERO is importable regardless of launch env (it lives outside this repo).
# Override with `LIBERO_SRC` env var on a new machine (e.g. LIBERO_SRC=D:/code/LIBERO).
_LIBERO_SRC = os.environ.get('LIBERO_SRC', 'C:/Users/DELL/.venv/LIBERO')
if os.path.isdir(_LIBERO_SRC) and _LIBERO_SRC not in sys.path:
    sys.path.insert(0, _LIBERO_SRC)
os.environ.setdefault('MUJOCO_GL', 'glfw')
import numpy as np
import torch
import cv2


def combined_frame(obs):
    """Side-by-side [agentview | wrist], flipped upright, with small labels.
    Shows the PERTURBED input (what the model actually sees) when PERTURB is active."""
    a = perturb_img(obs['agentview_image'], *PERTURB)[::-1]   # robosuite renders upside-down
    w = perturb_img(obs['robot0_eye_in_hand_image'], *PERTURB)[::-1]
    both = np.concatenate([a, w], axis=1).copy()  # [H, 2W, 3] RGB
    cv2.putText(both, 'agentview', (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    cv2.putText(both, 'wrist', (a.shape[1] + 4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return both


def write_video(frames, path, fps=20):
    """frames: list of HxWx3 uint8 RGB (already combined+upright). Saved as MP4."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torchvision import transforms
from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer

EVAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_proprio(obs):
    import robosuite.utils.transform_utils as T
    ee_pos = obs['robot0_eef_pos']                       # (3,)
    ee_ori = T.quat2axisangle(obs['robot0_eef_quat'])    # (3,) matches HDF5 ee_ori
    grip = obs['robot0_gripper_qpos']                    # (2,)
    joints = obs['robot0_joint_pos']                     # (7,)
    return np.concatenate([ee_pos, ee_ori, grip, joints]).astype(np.float32)  # 15D


# Robustness perturbations (applied to the raw uint8 RGB before resize/normalize).
# Severity-scaled, ImageNet-C-inspired. set by main() from CLI; (kind, severity).
PERTURB = ('none', 0)

def perturb_img(img, kind, sev):
    """img: HxWx3 uint8 RGB. Returns perturbed uint8."""
    if kind == 'none' or sev <= 0:
        return img
    im = img.astype(np.float32)
    if kind == 'noise':                       # gaussian pixel noise, std grows with severity
        std = [0, 5, 10, 16, 24, 34][min(sev, 5)]
        im = im + np.random.normal(0.0, std, im.shape)
    elif kind == 'blur':                      # gaussian blur, sigma grows with severity
        sigma = [0, 0.6, 1.1, 1.8, 2.7, 4.0][min(sev, 5)]
        kk = max(1, int(2 * round(sigma) + 1)) | 1
        im = cv2.GaussianBlur(im, (kk, kk), sigma)
    elif kind == 'brightness':                # additive brightness shift
        im = im + [0, 18, 36, 58, 84, 115][min(sev, 5)]
    return np.clip(im, 0, 255).astype(np.uint8)

def img6(obs, device):
    a = EVAL_TF(np.ascontiguousarray(perturb_img(obs['agentview_image'], *PERTURB)))
    w = EVAL_TF(np.ascontiguousarray(perturb_img(obs['robot0_eye_in_hand_image'], *PERTURB)))
    return torch.cat([a, w], 0).unsqueeze(0).to(device)


def target_obj_key(language, obs):
    """'pick up the alphabet soup and place it in the basket' -> 'alphabet_soup_1_pos'."""
    name = language.lower()
    if 'pick up the' in name:
        name = name.split('pick up the', 1)[1]
    if ' and place' in name:
        name = name.split(' and place', 1)[0]
    name = name.strip().replace(' ', '_')
    for k in obs:
        if k.startswith(name) and k.endswith('_pos') and 'to_robot' not in k:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='E:/fno_data/checkpoints/best.pt')
    ap.add_argument('--suite', default='libero_object')
    ap.add_argument('--n_tasks', type=int, default=-1)      # -1 = all
    ap.add_argument('--task_indices', default='')           # comma-sep idxs (e.g. 1,3,4,9); overrides n_tasks
    ap.add_argument('--output_size', type=int, default=0)   # 0=native; else decode FNO at this res, resample to exec_res (resolution-invariance test)
    ap.add_argument('--exec_res', type=int, default=16)     # execution chunk length (training chunk_size); decode resampled to this
    ap.add_argument('--n_rollouts', type=int, default=10)
    ap.add_argument('--perturb', default='none')         # robustness: none|noise|blur|brightness
    ap.add_argument('--severity', type=int, default=0)   # 0=clean .. 5=strongest
    ap.add_argument('--max_steps', type=int, default=400)
    ap.add_argument('--execute', type=int, default=8)
    ap.add_argument('--save_videos', action='store_true')
    ap.add_argument('--video_dir', default='E:/fno_data/eval_videos')
    ap.add_argument('--videos_per_task', type=int, default=2)
    ap.add_argument('--log_jsonl', default='')   # per-rollout structured log (z-error etc.)
    ap.add_argument('--tag', default='')         # free label stored in every record (e.g. exec8)
    args = ap.parse_args()
    global PERTURB
    PERTURB = (args.perturb, args.severity)
    np.random.seed(0)   # reproducible perturbation noise across runs
    if args.perturb != 'none' and args.severity > 0:
        print(f"[robustness] perturb={args.perturb} severity={args.severity}")
    if args.save_videos:
        os.makedirs(args.video_dir, exist_ok=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True

    ckpt = torch.load(args.checkpoint, map_location=dev)
    cfg = ckpt['config']
    amean = np.asarray(ckpt['action_mean'], np.float32)
    astd = np.asarray(ckpt['action_std'], np.float32)
    model = FNOVLA(cfg).to(dev); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    lc_cfg = cfg['model'].get('language', {})
    if lc_cfg.get('pretrained', False):
        from src.data.tokenizer import HFTokenizerAdapter
        tok = HFTokenizerAdapter(lc_cfg['model_name'], max_seq_len=lc_cfg['max_seq_len'])
    else:
        tok = SimpleTokenizer(max_seq_len=ckpt.get('tokenizer_max_seq_len', 64))
        tok.word2idx = ckpt['tokenizer_word2idx']
        tok.idx2word = {v: k for k, v in tok.word2idx.items()}

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()[args.suite]()
    if args.task_indices:
        task_list = [int(x) for x in args.task_indices.split(',') if x.strip() != '']
    else:
        nt = bench.get_num_tasks() if args.n_tasks < 0 else min(args.n_tasks, bench.get_num_tasks())
        task_list = list(range(nt))
    n_tasks = len(task_list)

    bddl_root = get_libero_path("bddl_files")
    init_root = get_libero_path("init_states")

    import json
    jlog = open(args.log_jsonl, 'a', encoding='utf-8') if args.log_jsonl else None

    results = {}
    t_start = time.time()
    for done_i, ti in enumerate(task_list):
        task = bench.get_task(ti)
        bddl = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        init_states = torch.load(os.path.join(init_root, task.problem_folder, task.init_states_file))

        tok_ids, attn = tok.encode(task.language)
        tok_ids = torch.tensor(tok_ids, dtype=torch.long, device=dev).unsqueeze(0)
        attn = torch.tensor(attn, dtype=torch.long, device=dev).unsqueeze(0)

        succ = 0
        task_jerks = []
        obj_key = None
        for r in range(args.n_rollouts):
            env.reset()
            init_idx = r % init_states.shape[0]
            obs = env.set_init_state(init_states[init_idx])
            for _ in range(5):  # settle physics
                obs, _, _, _ = env.step(np.zeros(7))
            if jlog is not None and obj_key is None:
                obj_key = target_obj_key(task.language, obs)
            record = args.save_videos and r < args.videos_per_task
            frames = [combined_frame(obs)] if record else None
            exec_actions = []   # executed continuous 6D commands -> jerk
            grasp_z_gap_min = float('inf')   # min |eef_z - obj_z| over approach
            min_eef_obj = float('inf'); grasp_dxy = None; grasp_dz = None
            max_lift = 0.0; min_basket_dxy = float('inf')
            obj_z0 = float(obs[obj_key][2]) if obj_key is not None else None
            done = False
            steps = 0
            while steps < args.max_steps and not done:
                with torch.no_grad():
                    out = model(img6(obs, dev), tok_ids,
                                torch.tensor(make_proprio(obs), device=dev).unsqueeze(0),
                                attention_mask=attn,
                                output_size=(args.output_size or None))
                acts = out['actions'][0].cpu().numpy() * astd + amean   # [H,6] real
                grip = out['gripper'][0, :, 0].cpu().numpy()            # [H]
                # resolution-invariance test: FNO decoded at output_size, resample
                # the trajectory back to exec_res so EXECUTION is identical across
                # resolutions (isolates decode-res effect from re-plan frequency).
                if args.output_size and acts.shape[0] != args.exec_res:
                    xi = np.linspace(0.0, 1.0, acts.shape[0])
                    xo = np.linspace(0.0, 1.0, args.exec_res)
                    acts = np.stack([np.interp(xo, xi, acts[:, c]) for c in range(acts.shape[1])], axis=1)
                    grip = np.interp(xo, xi, grip)
                for k in range(min(args.execute, acts.shape[0])):
                    g_act = 1.0 if grip[k] > 0.5 else -1.0
                    a = np.clip(np.concatenate([acts[k], [g_act]]), -1, 1)
                    exec_actions.append(a[:6].copy())
                    obs, _, _, info = env.step(a)
                    steps += 1
                    if obj_key is not None:
                        eef = np.asarray(obs['robot0_eef_pos'], float)
                        o = np.asarray(obs[obj_key], float)
                        dz = abs(float(eef[2]) - float(o[2]))
                        if dz < grasp_z_gap_min:
                            grasp_z_gap_min = dz
                        d3 = float(np.linalg.norm(eef - o))
                        if d3 < min_eef_obj:        # geometry at closest approach (grasp attempt)
                            min_eef_obj = d3
                            grasp_dxy = float(np.hypot(eef[0] - o[0], eef[1] - o[1]))
                            grasp_dz = float(eef[2] - o[2])     # +ve = gripper above object center
                        if obj_z0 is not None:
                            max_lift = max(max_lift, float(o[2]) - obj_z0)
                        if 'basket_1_pos' in obs:
                            b = np.asarray(obs['basket_1_pos'], float)
                            min_basket_dxy = min(min_basket_dxy, float(np.hypot(o[0] - b[0], o[1] - b[1])))
                    if record:
                        frames.append(combined_frame(obs))
                    if env.check_success():
                        done = True; break
            ok = env.check_success()
            succ += int(ok)
            # jerk = mean |2nd difference| of the executed continuous trajectory
            ea = np.asarray(exec_actions)
            jerk_r = float(np.abs(np.diff(ea, n=2, axis=0)).mean()) if ea.shape[0] >= 3 else float('nan')
            if ea.shape[0] >= 3:
                task_jerks.append(jerk_r)
            if jlog is not None:
                rec = {'tag': args.tag, 'execute': args.execute, 'task_idx': ti,
                       'task': task.language, 'rollout': r, 'init_idx': init_idx,
                       'perturb': args.perturb, 'severity': args.severity,
                       'success': int(ok), 'steps': steps, 'jerk': jerk_r,
                       'grasp_z_gap_min': (None if grasp_z_gap_min == float('inf') else grasp_z_gap_min),
                       'grasp_dxy': grasp_dxy, 'grasp_dz': grasp_dz, 'max_lift': max_lift,
                       'min_basket_dxy': (None if min_basket_dxy == float('inf') else min_basket_dxy)}
                if obj_key is not None and 'basket_1_pos' in obs:
                    op, bp = obs[obj_key], obs['basket_1_pos']
                    rec['final_obj_basket_dz'] = abs(float(op[2]) - float(bp[2]))
                    rec['final_obj_basket_dxy'] = float(np.hypot(op[0]-bp[0], op[1]-bp[1]))
                jlog.write(json.dumps(rec) + '\n'); jlog.flush()
            if record:
                tag = 'SUCCESS' if ok else 'FAIL'
                fn = f"{ti:02d}_{task.language.replace(' ', '_')[:40]}_r{r}_{tag}.mp4"
                write_video(frames, os.path.join(args.video_dir, fn))
        env.close()
        rate = succ / args.n_rollouts * 100
        jerk = float(np.mean(task_jerks)) if task_jerks else float('nan')
        results[task.language] = (rate, jerk)
        print(f"  [{done_i+1}/{n_tasks}] (task {ti}) {rate:5.1f}% ({succ}/{args.n_rollouts}) | jerk {jerk:.4f}  {task.language}")

    if jlog is not None:
        jlog.close()
        print(f"[z-error log] wrote per-rollout records to {args.log_jsonl}")

    rates = [v[0] for v in results.values()]
    jerks = [v[1] for v in results.values() if v[1] == v[1]]  # drop nan
    avg = float(np.mean(rates))
    mjerk = float(np.mean(jerks)) if jerks else float('nan')
    cfgm = ckpt['config']['model']['fno']
    print(f"\n=== {args.suite} | modes={cfgm['modes']}/chunk={cfgm['chunk_size']} | "
          f"success {avg:.1f}% | mean jerk {mjerk:.4f}  "
          f"({n_tasks}x{args.n_rollouts} rollouts, {(time.time()-t_start)/60:.1f} min) ===")


if __name__ == '__main__':
    main()
