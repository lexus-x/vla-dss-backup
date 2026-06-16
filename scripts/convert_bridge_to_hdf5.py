"""
Convert BridgeData V2 (RLDS / TFDS) -> FNO-VLA HDF5 schema, resized to the
training resolution so the on-disk size stays manageable.

Why convert (instead of reading RLDS in the training loop): it lets you reuse
the EXISTING lazy loader + model + training code unchanged. TensorFlow is then a
one-time conversion dependency, not part of the hot loop.

Output schema PER demo (matches LIBERODataset/LIBEROLazyDataset exactly, plus a
per-demo `language` attr since Bridge is per-trajectory NL, not per-file):
    data/demo_<i>/obs/agentview_rgb    [T,S,S,3] uint8   (Bridge external cam)
    data/demo_<i>/obs/eye_in_hand_rgb  [T,S,S,3] uint8   (DUPLICATE of external — Bridge has no wrist cam)
    data/demo_<i>/obs/ee_pos           [T,3] f32
    data/demo_<i>/obs/ee_ori           [T,3] f32
    data/demo_<i>/obs/gripper_states   [T,2] f32
    data/demo_<i>/obs/joint_states     [T,7] f32         (zeros — Bridge RLDS has no joints)
    data/demo_<i>/actions              [T,7] f32         (6 continuous + gripper; >0 == CLOSE)
    data/demo_<i>.attrs['language']    str

NOTE (verify on the A100 in Phase-1, RLDS key names vary by TFDS build):
  - external image key may be 'image_0' or 'image'
  - state layout is assumed [x,y,z, roll,pitch,yaw, gripper]; confirm against a sample
  - GRIPPER SIGN is the classic bug: set --gripper_close_is_one to match Bridge's
    convention so that stored actions[:,6] > 0 means CLOSE (your codebase's rule).

Usage:
    python scripts/convert_bridge_to_hdf5.py \
        --rlds_dir /workspace/tfds --dataset bridge_dataset \
        --out_dir /workspace/fno_data/bridge --img_size 128 --shard_size 200
"""
import argparse, os, json
import numpy as np
import h5py
import cv2  # resize


def _resize(img, s):
    return cv2.resize(np.asarray(img), (s, s), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rlds_dir', required=True, help='tfds data_dir')
    ap.add_argument('--dataset', default='bridge_dataset', help='tfds name (bridge_dataset or bridge)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--img_size', type=int, default=128)
    ap.add_argument('--shard_size', type=int, default=200, help='demos per HDF5 shard')
    ap.add_argument('--image_key', default='image_0')
    ap.add_argument('--gripper_close_is_one', action='store_true',
                    help='set if Bridge gripper action==1 means CLOSE; we remap so stored >0 == close')
    ap.add_argument('--clip_pct', type=float, default=1.0,
                    help='percentile clip for continuous actions (robust norm); 1.0 => clip to [1,99]')
    ap.add_argument('--max_episodes', type=int, default=-1, help='cap for quick subset runs')
    ap.add_argument('--tasks_file', default=None,
                    help='text file, one instruction/line; keep ONLY these (exact match) -> curated subset')
    ap.add_argument('--skill_keywords', default=None,
                    help='comma-separated substrings; keep episodes whose instruction contains any (e.g. "put,place,pick")')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tasks_set = None
    if args.tasks_file:
        with open(args.tasks_file) as f:
            tasks_set = set(l.strip() for l in f if l.strip())
        print(f'task filter: keeping {len(tasks_set)} selected instructions')
    skill_kw = [k.strip().lower() for k in args.skill_keywords.split(',')] if args.skill_keywords else None

    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # keep TF off the GPU during conversion

    ds = tfds.load(args.dataset, data_dir=args.rlds_dir, split='train')

    # ---- pass 1: sample continuous-action percentiles for robust clipping ----
    # Respect the task filter so norm stats match the CURATED slice, not the global set.
    s = args.img_size
    sample_actions = []
    n_seen = 0
    for ep in tfds.as_numpy(ds):
        lang = ''
        for st in ep['steps']:
            l = st.get('language_instruction', b'')
            l = l.decode() if isinstance(l, bytes) else str(l)
            if l.strip():
                lang = l.strip(); break
        if tasks_set is not None and lang not in tasks_set:
            continue
        if skill_kw and not any(k in lang.lower() for k in skill_kw):
            continue
        for st in ep['steps']:
            sample_actions.append(st['action'][:6])
        n_seen += 1
        if n_seen >= 2000:
            break
    sample_actions = np.asarray(sample_actions, dtype=np.float64)
    lo = np.percentile(sample_actions, args.clip_pct, axis=0)
    hi = np.percentile(sample_actions, 100 - args.clip_pct, axis=0)
    print('action clip lo', lo, 'hi', hi)

    # ---- pass 2: write shards ----
    shard, demo_in_shard, n_demos = None, 0, 0
    shard_idx = 0

    def open_shard(k):
        return h5py.File(os.path.join(args.out_dir, f'bridge_shard_{k:04d}.hdf5'), 'w')

    it = tfds.as_numpy(ds)
    for ep_i, ep in enumerate(it):
        if args.max_episodes > 0 and ep_i >= args.max_episodes:
            break
        imgs, eepos, eeori, grip2, acts, langs = [], [], [], [], [], []
        for st in ep['steps']:
            obs = st['observation']
            img = _resize(obs[args.image_key], s)
            state = np.asarray(obs['state'], dtype=np.float32)  # assumed [xyz,rpy,grip]
            a = np.asarray(st['action'], dtype=np.float32)      # [7]
            # remap gripper so >0 == close
            g = a[6]
            a6 = g - 0.5 if args.gripper_close_is_one else 0.5 - g
            a = np.concatenate([np.clip(a[:6], lo, hi), [a6]]).astype(np.float32)
            imgs.append(img)
            eepos.append(state[:3]); eeori.append(state[3:6])
            grip2.append([state[6], state[6]] if state.shape[0] > 6 else [0.0, 0.0])
            acts.append(a)
            lang = st.get('language_instruction', b'')
            langs.append(lang.decode() if isinstance(lang, bytes) else str(lang))
        imgs = np.stack(imgs).astype(np.uint8)           # [T,s,s,3]
        T = imgs.shape[0]
        eepos = np.stack(eepos).astype(np.float32)
        eeori = np.stack(eeori).astype(np.float32)
        grip2 = np.asarray(grip2, dtype=np.float32)
        acts = np.stack(acts).astype(np.float32)
        language = max(set(langs), key=langs.count) if langs else ''  # episode instruction
        if not language.strip() or T < 2:
            continue  # skip empty-language / degenerate trajectories
        # curated-subset filters (specialize the data)
        if tasks_set is not None and language not in tasks_set:
            continue
        if skill_kw and not any(k in language.lower() for k in skill_kw):
            continue
        # (aux grasp_target = eef x-y at gripper-close is recomputed by the loader
        #  from ee_pos + actions, so nothing extra to store here.)

        if shard is None or demo_in_shard >= args.shard_size:
            if shard is not None:
                shard.close(); shard_idx += 1
            shard = open_shard(shard_idx); demo_in_shard = 0
        g = shard.create_group(f'data/demo_{demo_in_shard}')
        o = g.create_group('obs')
        o.create_dataset('agentview_rgb', data=imgs, compression='gzip', compression_opts=4)
        o.create_dataset('eye_in_hand_rgb', data=imgs, compression='gzip', compression_opts=4)  # dup (no wrist)
        o.create_dataset('ee_pos', data=eepos)
        o.create_dataset('ee_ori', data=eeori)
        o.create_dataset('gripper_states', data=grip2)
        o.create_dataset('joint_states', data=np.zeros((T, 7), np.float32))  # Bridge has no joints
        g.create_dataset('actions', data=acts)
        g.attrs['language'] = language
        demo_in_shard += 1; n_demos += 1
        if n_demos % 500 == 0:
            print(f'{n_demos} demos written...')

    if shard is not None:
        shard.close()
    with open(os.path.join(args.out_dir, 'action_clip.json'), 'w') as f:
        json.dump({'lo': lo.tolist(), 'hi': hi.tolist(), 'img_size': s}, f, indent=2)
    print(f'DONE: {n_demos} demos in {shard_idx + 1} shards -> {args.out_dir}')


if __name__ == '__main__':
    main()
