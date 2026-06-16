"""
DAgger data collection for FNO-VLA (x-y fix #3).

Rolls out the CURRENT policy in LIBERO-Object (it visits its own state
distribution, including the off-center approach states that the expert demos
never covered), and at every executed step records:
    obs (agentview + wrist RGB, proprio)         <- the policy-visited STATE
    oracle corrective action from privileged pose <- the EXPERT LABEL

The env is stepped with the POLICY's action (β=0 DAgger), so we aggregate
(policy-visited-state -> expert-action) pairs. The privileged object pose is used
ONLY to compute the label, never fed to the model -> no train/test mismatch.

Writes the exact HDF5 schema the trainer reads (same as generate_grasp_demos.py):
  data/demo_{k}/obs/{agentview_rgb,eye_in_hand_rgb,ee_pos,ee_ori,gripper_states,joint_states}
  data/demo_{k}/actions  [T,7]   <- ORACLE actions

Run (after the GPU is free):
  python scripts/collect_dagger.py --checkpoint E:/fno_data/run_dinov3_auxxy/epoch_5.pt \
      --task_indices 3,4,0,5,8,9 --rollouts_per_task 12 --out E:/fno_data/dagger/dagger_auxxy.h5
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('MUJOCO_GL', 'glfw')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch
import h5py
# reuse the EXACT preprocessing the eval/policy uses
from eval_sim import make_proprio, img6, target_obj_key, FNOVLA, SimpleTokenizer


def oracle_action(obs, obj_key, obj_z0, gain=15.0, grasp_z=0.0):
    """Stateless privileged-pose corrective action (7D: dpos, 0,0,0, grip).

    Phase is inferred from geometry so it is valid at ANY policy-visited state:
      not grasped & off in xy  -> move to TRUE object xy  (the x-y correction)
      not grasped & aligned hi -> descend to grasp height
      not grasped & aligned lo -> close
      grasped                  -> carry to basket, then release
    """
    eef = np.asarray(obs['robot0_eef_pos'], float)
    obj = np.asarray(obs[obj_key], float)
    basket = np.asarray(obs['basket_1_pos'], float)
    obj_lifted = (obj[2] - obj_z0) > 0.04
    dxy = float(np.linalg.norm(eef[:2] - obj[:2]))
    if obj_lifted:                                   # have the object -> carry / release
        over_basket = np.linalg.norm(eef[:2] - basket[:2]) < 0.04
        tgt = np.array([basket[0], basket[1], basket[2] + 0.20])
        grip = -1.0 if over_basket else 1.0
    else:
        if dxy > 0.012:                              # off in xy -> go above TRUE obj xy
            tgt = np.array([obj[0], obj[1], obj[2] + 0.10]); grip = -1.0
        elif eef[2] - (obj[2] + grasp_z) > 0.015:    # aligned, high -> descend
            tgt = np.array([obj[0], obj[1], obj[2] + grasp_z]); grip = -1.0
        else:                                        # aligned, low -> close
            tgt = eef.copy(); grip = 1.0
    dpos = np.clip((tgt - eef) * gain, -1.0, 1.0)
    return np.concatenate([dpos, np.zeros(3), [grip]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='E:/fno_data/run_dinov3_auxxy/epoch_5.pt')
    ap.add_argument('--suite', default='libero_object')
    ap.add_argument('--task_indices', default='3,4,0,5,8,9')  # default: the off-center losers
    ap.add_argument('--rollouts_per_task', type=int, default=12)
    ap.add_argument('--execute', type=int, default=8)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--gain', type=float, default=15.0)
    ap.add_argument('--grasp_z', type=float, default=0.0)
    ap.add_argument('--keep', default='all', choices=['all', 'fail', 'divergent'])
    ap.add_argument('--div_thresh', type=float, default=0.3)  # |oracle-policy| dpos to count as divergent
    # per-task dagger files land in <out_dir>/<suite>/<lang>_dagger_demo.hdf5 so the
    # trainer (which derives language from the filename) labels them correctly.
    ap.add_argument('--out_dir', default='E:/fno_data/libero_object_dagger')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpt = torch.load(args.checkpoint, map_location=dev)
    cfg = ckpt['config']
    amean = np.asarray(ckpt['action_mean'], np.float32)
    astd = np.asarray(ckpt['action_std'], np.float32)
    model = FNOVLA(cfg).to(dev); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    print(f"[dagger] loaded {args.checkpoint} (epoch {ckpt['epoch']})", flush=True)

    tok = SimpleTokenizer(max_seq_len=ckpt.get('tokenizer_max_seq_len', 64))
    tok.word2idx = ckpt['tokenizer_word2idx']
    tok.idx2word = {v: k for k, v in tok.word2idx.items()}

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()[args.suite]()
    bddl_root = get_libero_path('bddl_files'); init_root = get_libero_path('init_states')
    task_list = [int(x) for x in args.task_indices.split(',') if x.strip() != '']

    suite_out = os.path.join(args.out_dir, args.suite)
    os.makedirs(suite_out, exist_ok=True)
    total_demos = 0; total_pairs = 0
    for ti in task_list:
        task = bench.get_task(ti)
        demos = []   # per-task demos in trainer schema
        n_pairs = 0
        bddl = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        init_states = torch.load(os.path.join(init_root, task.problem_folder, task.init_states_file))
        tok_ids, attn = tok.encode(task.language)
        tok_ids = torch.tensor(tok_ids, dtype=torch.long, device=dev).unsqueeze(0)
        attn = torch.tensor(attn, dtype=torch.long, device=dev).unsqueeze(0)
        obj_key = None
        kept_here = 0
        for r in range(args.rollouts_per_task):
            env.reset()
            init_idx = r % init_states.shape[0]
            obs = env.set_init_state(init_states[init_idx])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))
            if obj_key is None:
                obj_key = target_obj_key(task.language, obs)
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
                    # EXPERT label at the current (policy-visited) state
                    a_oracle = oracle_action(obs, obj_key, obj_z0, args.gain, args.grasp_z)
                    # POLICY action that actually drives the env (β=0 DAgger)
                    g_act = 1.0 if grip[k] > 0.5 else -1.0
                    a_policy = np.clip(np.concatenate([acts[k], [g_act]]), -1, 1).astype(np.float32)
                    max_div = max(max_div, float(np.abs(a_oracle[:3] - a_policy[:3]).max()))
                    # record STATE (obs) + oracle LABEL, matching HDF5 schema (flipped RGB)
                    rec['agentview_rgb'].append(obs['agentview_image'][::-1])
                    rec['eye_in_hand_rgb'].append(obs['robot0_eye_in_hand_image'][::-1])
                    rec['ee_pos'].append(np.asarray(obs['robot0_eef_pos'], np.float32))
                    import robosuite.utils.transform_utils as T
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
        # write ONE dagger file per task, named so the trainer derives the right language
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
        print(f"=== task {ti} ({task.language[:30]}): kept {kept_here}/{args.rollouts_per_task}, "
              f"{len(demos)} demos / {n_pairs} pairs -> {fname} ===", flush=True)

    print(f"\n[dagger] TOTAL {total_demos} demos, {total_pairs} state-action pairs in {suite_out}", flush=True)


if __name__ == '__main__':
    main()
