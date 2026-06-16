"""
List unique Bridge instructions + counts so you can pick a CURATED subset
(the SmolVLA-validated lever). Run this on the A100 after the RLDS is downloaded.

Output:
  * prints a skill-keyword breakdown + the top-N instructions by frequency
  * writes bridge_tasks.json (full ranked list)
Then edit the printed list down to the instructions you want and save them
one-per-line to e.g. selected_tasks.txt, and pass that to the converter:
  python scripts/convert_bridge_to_hdf5.py ... --tasks_file selected_tasks.txt

Usage:
  python scripts/list_bridge_tasks.py --rlds_dir /workspace/tfds --dataset bridge_dataset
"""
import argparse, json, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rlds_dir', required=True)
    ap.add_argument('--dataset', default='bridge_dataset')
    ap.add_argument('--out', default='bridge_tasks.json')
    ap.add_argument('--top', type=int, default=80)
    args = ap.parse_args()

    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    ds = tfds.load(args.dataset, data_dir=args.rlds_dir, split='train')
    counts = collections.Counter()
    n_ep = 0
    for ep in tfds.as_numpy(ds):
        lang = ''
        for st in ep['steps']:
            l = st.get('language_instruction', b'')
            l = l.decode() if isinstance(l, bytes) else str(l)
            if l.strip():
                lang = l.strip()
                break
        if lang:
            counts[lang] += 1
        n_ep += 1

    # crude skill bucketing by leading action verb
    SKILLS = ['put', 'place', 'pick', 'push', 'move', 'open', 'close',
              'stack', 'fold', 'wipe', 'sweep', 'turn', 'flip', 'press']
    skill_counts = collections.Counter()
    for inst, c in counts.items():
        words = inst.lower().split()
        verb = next((s for s in SKILLS if s in words), 'other')
        skill_counts[verb] += c

    print(f'episodes: {n_ep}   unique instructions: {len(counts)}')
    print('--- skill buckets (by verb) ---')
    for s, c in skill_counts.most_common():
        print(f'  {s:8s} {c}')
    print(f'--- top {args.top} instructions ---')
    for inst, c in counts.most_common(args.top):
        print(f'  {c:5d}  {inst}')
    with open(args.out, 'w') as f:
        json.dump({'n_episodes': n_ep,
                   'instructions': counts.most_common(),
                   'skills': skill_counts.most_common()}, f, indent=2)
    print(f'wrote {args.out} -> edit down to your selected list (one instruction/line)')


if __name__ == '__main__':
    main()
