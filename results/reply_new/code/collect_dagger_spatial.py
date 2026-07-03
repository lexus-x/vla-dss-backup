"""
Hand-crafted DAgger collection for LIBERO-Spatial (mirrors collect_dagger.py for
Object, which gave +8.5pp). The generalized oracle failed Spatial (66% < 73%)
because target_obj_key fuzzy-matches the language ("black bowl between the plate
and the ramekin") and never resolves the real body. Here we PARSE THE BDDL GOAL
predicate `(On <obj> <region>)` -> exact target body + place target. For every
Spatial task the goal is `(On akita_black_bowl_1 plate_1)`.

Rolls out the CURRENT policy (beta=0 DAgger: env stepped by policy action), and at
every executed step records (policy-visited obs) -> (privileged-pose oracle action).
Place target is the PLATE (place-on, not drop-in-basket).

Writes the trainer HDF5 schema, one file per task named so the trainer derives the
language from the filename.
"""
import os, sys, argparse, re
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.environ.get('LIBERO_SRC', 'C:/code/LIBERO'))
os.environ.setdefault('MUJOCO_GL', 'glfw')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch
import h5py
from eval_sim import make_proprio, img6, FNOVLA, SimpleTokenizer


def parse_goal(bddl_path):
    """(On akita_black_bowl_1 plate_1) -> ('akita_black_bowl_1_pos','plate_1_pos')."""
    txt = open(bddl_path).read()
    idx = txt.find('(:goal')
    blk = txt[idx:idx + 300] if idx >= 0 else ''
    m = re.search(r'\(\s*On\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)', blk)
    if not m:
        return 'akita_black_bowl_1_pos', 'plate_1_pos'
    return m.group(1) + '_pos', m.group(2) + '_pos'


def oracle_action(obs, obj_key, plate_key, obj_z0, gain=15.0, grasp_z=0.0):
    """Privileged-pose corrective action (7D). Phase inferred from geometry so it
    is valid at ANY policy-visited state:
      not lifted & off xy   -> move above TRUE bowl xy  (the x-y correction)
      not lifted & aligned  -> descend to grasp height, then close
      lifted                -> carry above plate, release when over plate
    """
    eef = np.asarray(obs['robot0_eef_pos'], float)
    obj = np.asarray(obs[obj_key], float)
    plate = np.asarray(obs[plate_key], float)
    obj_lifted = (obj[2] - obj_z0) > 0.03            # bowls are light/low -> smaller thresh
    dxy = float(np.linalg.norm(eef[:2] - obj[:2]))
    if obj_lifted:                                    # have the bowl -> carry / release over plate
        over_plate = np.linalg.norm(eef[:2] - plate[:2]) < 0.04
        tgt = np.array([plate[0], plate[1], plate[2] + 0.12])
        grip = -1.0 if over_plate else 1.0
    else:
        if dxy > 0.012:                               # off in xy -> above TRUE bowl xy
            tgt = np.array([obj[0], obj[1], obj[2] + 0.10]); grip = -1.0
        elif eef[2] - (obj[2] + grasp_z) > 0.015:     # aligned, high -> descend
            tgt = np.array([obj[0], obj[1], obj[2] + grasp_z]); grip = -1.0
        else:                                         # aligned, low -> close
            tgt = eef.copy(); grip = 1.0
    dpos = np.clip((tgt - eef) * gain, -1.0, 1.0)
    return np.concatenate([dpos, np.zeros(3), [grip]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='E:/fno_data/run_dinov3_spatial_aux/epoch_15.pt')
    ap.add_argument('--suite', default='libero_spatial')
    ap.add_argument('--task_indices', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--rollouts_per_task', type=int, default=15)
    ap.add_argument('--execute', type=int, default=8)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--gain', type=float, default=15.0)
    ap.add_argument('--grasp_z', type=float, default=0.0)
    ap.add_argument('--keep', default='all', choices=['all', 'fail', 'divergent'])
    ap.add_argument('--div_thresh', type=float, default=0.3)
    ap.add_argument('--out_dir', default='E:/fno_data/libero_spatial_dagger')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpt = torch.load(args.checkpoint, map_location=dev)
    cfg = ckpt['config']
    amean = np.asarray(ckpt['action_mean'], np.float32)
    astd = np.asarray(ckpt['action_std'], np.float32)
    model = FNOVLA(cfg).to(dev); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    print(f"[dagger-sp] loaded {args.checkpoint} (epoch {ckpt['epoch']})", flush=True)

    tok = SimpleTokenizer(max_seq_len=ckpt.get('tokenizer_max_seq_len', 64))
    tok.word2idx = ckpt['tokenizer_word2idx']
    tok.idx2word = {v: k for k, v in tok.word2idx.items()}

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.transform_utils as T
    bench = benchmark.get_benchmark_dict()[args.suite]()
    bddl_root = get_libero_path('bddl_files'); init_root = get_libero_path('init_states')
    task_list = [int(x) for x in args.task_indices.split(',') if x.strip() != '']

    suite_out = os.path.join(args.out_dir, args.suite)
    os.makedirs(suite_out, exist_ok=True)
    total_demos = 0; total_pairs = 0
    for ti in task_list:
        task = bench.get_task(ti)
        bddl = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        obj_key, plate_key = parse_goal(bddl)
        demos = []; n_pairs = 0
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        init_states = torch.load(os.path.join(init_root, task.problem_folder, task.init_states_file))
        tok_ids, attn = tok.encode(task.language)
        tok_ids = torch.tensor(tok_ids, dtype=torch.long, device=dev).unsqueeze(0)
        attn = torch.tensor(attn, dtype=torch.long, device=dev).unsqueeze(0)
        kept_here = 0
        for r in range(args.rollouts_per_task):
            env.reset()
            obs = env.set_init_state(init_states[r % init_states.shape[0]])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))
            if obj_key not in obs or plate_key not in obs:
                print(f"  [warn] task {ti}: {obj_key}/{plate_key} not in obs; keys="
                      f"{[k for k in obs if k.endswith('_pos')][:6]}", flush=True)
            obj_z0 = float(obs[obj_key][2])
            rec = {k: [] for k in ['agentview_rgb', 'eye_in_hand_rgb', 'ee_pos', 'ee_ori',
                                   'gripper_states', 'joint_states', 'actions']}
            steps = 0; done = False; max_div = 0.0
            while steps < args.max_steps and not done:
                with torch.no_grad():
                    out = model(img6(obs, dev), tok_ids,
                                torch.tensor(make_proprio(obs), device=dev).unsqueeze(0),
                                attention_mask=attn)
                acts = out['actions'][0].cpu().numpy() * astd + amean
                grip = out['gripper'][0, :, 0].cpu().numpy()
                for k in range(min(args.execute, acts.shape[0])):
                    a_oracle = oracle_action(obs, obj_key, plate_key, obj_z0, args.gain, args.grasp_z)
                    g_act = 1.0 if grip[k] > 0.5 else -1.0
                    a_policy = np.clip(np.concatenate([acts[k], [g_act]]), -1, 1).astype(np.float32)
                    max_div = max(max_div, float(np.abs(a_oracle[:3] - a_policy[:3]).max()))
                    rec['agentview_rgb'].append(obs['agentview_image'][::-1])
                    rec['eye_in_hand_rgb'].append(obs['robot0_eye_in_hand_image'][::-1])
                    rec['ee_pos'].append(np.asarray(obs['robot0_eef_pos'], np.float32))
                    rec['ee_ori'].append(T.quat2axisangle(obs['robot0_eef_quat']).astype(np.float32))
                    rec['gripper_states'].append(np.asarray(obs['robot0_gripper_qpos'], np.float32))
                    rec['joint_states'].append(np.asarray(obs['robot0_joint_pos'], np.float32))
                    rec['actions'].append(a_oracle)
                    obs, _, done, _ = env.step(a_policy)
                    steps += 1
                    if env.check_success():
                        done = True; break
            ok = env.check_success()
            keep = (args.keep == 'all' or (args.keep == 'fail' and not ok)
                    or (args.keep == 'divergent' and max_div >= args.div_thresh))
            if keep and len(rec['actions']) >= 3:
                demos.append({k: np.asarray(v) for k, v in rec.items()})
                n_pairs += len(rec['actions']); kept_here += 1
            print(f"  task {ti} r{r:2d}: {'OK ' if ok else 'fail'} steps={steps} "
                  f"maxdiv={max_div:.2f} kept={keep}", flush=True)
        env.close()
        fname = task.language.replace(' ', '_') + '_dagger_demo.hdf5'
        fpath = os.path.join(suite_out, fname)
        with h5py.File(fpath, 'w') as f:
            g = f.create_group('data')
            for k, tr in enumerate(demos):
                dg = g.create_group(f'demo_{k}'); og = dg.create_group('obs')
                og.create_dataset('agentview_rgb', data=tr['agentview_rgb'].astype(np.uint8))
                og.create_dataset('eye_in_hand_rgb', data=tr['eye_in_hand_rgb'].astype(np.uint8))
                og.create_dataset('ee_pos', data=tr['ee_pos'])
                og.create_dataset('ee_ori', data=tr['ee_ori'])
                og.create_dataset('gripper_states', data=tr['gripper_states'])
                og.create_dataset('joint_states', data=tr['joint_states'])
                dg.create_dataset('actions', data=tr['actions'])
        total_demos += len(demos); total_pairs += n_pairs
        print(f"=== task {ti} ({task.language[:40]}) obj={obj_key} plate={plate_key}: "
              f"kept {kept_here}/{args.rollouts_per_task}, {len(demos)} demos / {n_pairs} pairs ===", flush=True)

    print(f"\n[dagger-sp] TOTAL {total_demos} demos, {total_pairs} pairs in {suite_out}", flush=True)


if __name__ == '__main__':
    main()
