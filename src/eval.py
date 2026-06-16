"""
Evaluation script for FNO-VLA.
Runs rollouts in LIBERO environment and reports success rate.
"""

import os
import sys
import math
import argparse
import yaml

import torch
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import FNOVLA
from src.data.tokenizer import SimpleTokenizer

# Image transform for evaluation (no augmentation)
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def run_rollout(
    model: FNOVLA,
    env,
    task_name: str,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    max_steps: int = 500,
    execute_steps: int = 4,
    output_size: int | None = None,
    temporal_ensemble: bool = True,
     te_m: float = 0.1,
) -> tuple[bool, int]:
    """
    Run a single rollout in the LIBERO environment.

    If temporal_ensemble is True, the policy is queried every timestep and the
    overlapping action-chunk predictions for each timestep are combined with
    exponential weights (ACT / OpenVLA-OFT style), which yields smoother, more
    robust rollouts than open-loop chunk execution.

    Returns (success, num_steps).
    """
    model.eval()
    obs = env.reset()

    token_ids, attn_mask = tokenizer.encode(task_name)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    attn_mask = torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0)

    def predict(obs):
        # Two views stacked on channels -> [1, 6, H, W] (matches training)
        img = EVAL_TRANSFORM(obs['agentview_rgb'])
        wrist = EVAL_TRANSFORM(obs['eye_in_hand_rgb'])
        image_tensor = torch.cat([img, wrist], dim=0).unsqueeze(0).to(device)
        proprio = np.concatenate([
            obs['ee_pos'], obs['ee_ori'], obs['gripper_states'], obs['joint_states'],
        ])  # 15D
        proprio_tensor = torch.tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0)
        out = model(image_tensor, token_ids, proprio_tensor,
                    attention_mask=attn_mask, output_size=output_size)
        actions = out['actions'][0].cpu().numpy()        # [chunk, 6] (normalized)
        if action_mean is not None and action_std is not None:
            actions = actions * action_std + action_mean  # un-normalize
        gripper = out['gripper'][0, :, 0].cpu().numpy()  # [chunk] per-step probs
        return actions, gripper

    def step_env(continuous6, gripper_prob):
        gripper_action = 1.0 if gripper_prob > 0.5 else -1.0
        full_action = np.clip(np.concatenate([continuous6, [gripper_action]]), -1.0, 1.0)
        return env.step(full_action)

    total_steps = 0
    success = False

    with torch.no_grad():
        if not temporal_ensemble:
            # Open-loop chunk execution (the original behaviour, kept for ablation)
            while total_steps < max_steps:
                actions, gripper = predict(obs)  # gripper: [chunk] per-step
                for k in range(min(execute_steps, actions.shape[0])):
                    obs, reward, done, info = step_env(actions[k], gripper[k])
                    total_steps += 1
                    if done or info.get('success', False):
                        return info.get('success', False), total_steps
                    if total_steps >= max_steps:
                        break
            return success, total_steps

        # Temporal ensembling: query every step, exp-weighted average of all
        # predictions that target the current timestep.
        preds = []  # list of (start_t, actions[H,6], gripper[H])
        t = 0
        while t < max_steps:
            actions, gripper = predict(obs)
            preds.append((t, actions, gripper))
            preds = [p for p in preds if p[0] + p[1].shape[0] > t]  # drop stale

            acc = np.zeros(6, dtype=np.float64)
            gacc = 0.0
            wsum = 0.0
            for start_t, act, gp in preds:
                idx = t - start_t
                if 0 <= idx < act.shape[0]:
                    w = math.exp(-te_m * idx)  # newer predictions weighted higher
                    acc += w * act[idx]
                    gacc += w * gp[idx]
                    wsum += w
            continuous6 = acc / max(wsum, 1e-8)
            gripper_prob = gacc / max(wsum, 1e-8)

            obs, reward, done, info = step_env(continuous6, gripper_prob)
            t += 1
            total_steps = t
            if done or info.get('success', False):
                return info.get('success', False), total_steps

    return success, total_steps


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    suite: str = "libero_object",
    num_rollouts: int = 20,
    output_size: int | None = None,
):
    """Evaluate model on LIBERO benchmark."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']

    model = FNOVLA(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Action normalization stats (model outputs are normalized; un-normalize)
    action_mean = ckpt.get('action_mean')
    action_std = ckpt.get('action_std')
    if action_mean is not None:
        print(f"Using action normalization from checkpoint (mean/std present)")

    # Rebuild tokenizer from the SAVED training vocab so word->id mapping matches
    # what the language embeddings were trained with. Re-fitting here (the old
    # behaviour) produced a different vocab and broke language conditioning.
    tokenizer = SimpleTokenizer(max_seq_len=ckpt.get('tokenizer_max_seq_len', 64))
    if 'tokenizer_word2idx' in ckpt:
        tokenizer.word2idx = ckpt['tokenizer_word2idx']
        tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}
        tokenizer._next_idx = len(tokenizer.word2idx)
    else:
        print("WARNING: checkpoint has no tokenizer vocab; language conditioning "
              "may not match training.")

    # Import LIBERO
    from libero.libero import benchmark
    bench_cls = benchmark.get_benchmark_dict()[suite]
    bench = bench_cls()

    task_names = bench.get_task_names()
    if 'tokenizer_word2idx' not in ckpt:
        tokenizer.fit(task_names)

    print(f"\nEvaluating on {suite} ({bench.get_num_tasks()} tasks, {num_rollouts} rollouts each)")
    if output_size is not None:
        print(f"Resolution-invariance test: output_size={output_size}")

    # Run evaluation
    results = {}
    for task_idx in range(bench.get_num_tasks()):
        task_name = task_names[task_idx]

        # Create environment
        task = bench.get_task(task_idx)
        env = task.create_env()

        successes = 0
        for rollout_idx in range(num_rollouts):
            success, steps = run_rollout(
                model, env, task_name, tokenizer, device,
                action_mean=action_mean, action_std=action_std,
                output_size=output_size,
            )
            successes += int(success)

        success_rate = successes / num_rollouts * 100
        results[task_name] = success_rate
        print(f"  {task_name}: {success_rate:.0f}% ({successes}/{num_rollouts})")

        env.close()

    avg = np.mean(list(results.values()))
    print(f"\nAverage success rate: {avg:.1f}%")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='/home/sarvik/Desktop/VLA-main/LIBERO/datasets')
    parser.add_argument('--suite', default='libero_object')
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--output_size', type=int, default=None,
                        help='Override action chunk size (resolution-invariance test)')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.suite, args.num_rollouts, args.output_size)
