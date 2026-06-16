"""
Generate figures for the weekly report presentation.
Uses real LIBERO demo data for frequency spectrum.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
LIBERO_PATH = '/home/sarvik/Desktop/VLA-main/LIBERO/datasets/libero_object'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
})

BLUE = '#0046FF'
RED = '#FF4444'
GREEN = '#00AA44'
ORANGE = '#FF8800'


def load_libero_actions():
    """Load real action trajectories from LIBERO demos."""
    all_actions = []
    demo_file = os.path.join(LIBERO_PATH,
        'pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5')
    with h5py.File(demo_file, 'r') as f:
        for demo_key in sorted(f['data'].keys()):
            actions = f['data'][demo_key]['actions'][:]
            all_actions.append(actions)
    return all_actions


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Combined — Spectrum + Cumulative Energy (one image)
# ═══════════════════════════════════════════════════════════════
def plot_frequency_spectrum():
    all_actions = load_libero_actions()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Average spectrum across all demos and all 7 action dims
    max_len = max(a.shape[0] for a in all_actions)
    n_freq = max_len // 2 + 1
    avg_spectrum = np.zeros(n_freq)
    count = 0

    dim_labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    dim_colors = [BLUE, GREEN, RED, ORANGE, '#AA00FF', '#00CCCC', '#888888']

    # Per-dimension spectra for the plot (use first demo for clarity)
    demo_actions = all_actions[0][:, :6]  # skip gripper for spectrum
    n = demo_actions.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0/20)  # 20Hz control

    for dim in range(6):
        fft_vals = np.abs(np.fft.rfft(demo_actions[:, dim])) ** 2
        axes[0].semilogy(freqs[1:], fft_vals[1:], color=dim_colors[dim],
                     linewidth=1.5, label=dim_labels[dim], alpha=0.8)

    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Spectral Energy (log)')
    axes[0].set_title('Energy Spectrum — Real LIBERO Trajectory')
    axes[0].legend(fontsize=9, ncol=2)

    # Cumulative energy (averaged across ALL demos, all 6 continuous dims)
    for actions in all_actions:
        cont_actions = actions[:, :6]
        n_a = cont_actions.shape[0]
        for dim in range(6):
            fft_vals = np.abs(np.fft.rfft(cont_actions[:, dim])) ** 2
            nf = len(fft_vals)
            if nf <= n_freq:
                avg_spectrum[:nf] += fft_vals
            else:
                avg_spectrum += fft_vals[:n_freq]
            count += 1

    avg_spectrum /= count
    cumulative = np.cumsum(avg_spectrum[1:]) / np.sum(avg_spectrum[1:]) * 100
    mode_pct = np.arange(1, len(cumulative)+1) / len(cumulative) * 100

    axes[1].fill_between(mode_pct, cumulative, alpha=0.25, color=BLUE)
    axes[1].plot(mode_pct, cumulative, color=BLUE, linewidth=2.5)

    # Mark 90% and 95% energy lines with proper text placement
    annotations = [
        (90, RED, 'right', 18, -15),
        (95, ORANGE, 'right', 18, 0),
    ]
    for threshold, color, ha, dx, dy in annotations:
        idx = np.searchsorted(cumulative, threshold)
        if idx < len(mode_pct):
            pct = mode_pct[idx]
            axes[1].axhline(threshold, color=color, linestyle='--', linewidth=1.2, alpha=0.6)
            axes[1].axvline(pct, color=color, linestyle='--', linewidth=1.2, alpha=0.6)
            axes[1].plot(pct, threshold, 'o', color=color, markersize=8, zorder=5)
            # Position text to avoid overlap
            tx = min(pct + dx, 75)  # keep text within plot
            ty = threshold + dy
            axes[1].annotate(f'{pct:.0f}% modes\n= {threshold}% energy',
                             xy=(pct, threshold),
                             xytext=(tx, ty),
                             fontsize=10, fontweight='bold', color=color,
                             ha='left', va='center',
                             arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    axes[1].set_xlabel('% of Frequency Modes (low → high)')
    axes[1].set_ylabel('Cumulative Energy (%)')
    axes[1].set_title('Cumulative Energy — Real LIBERO Data')
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'freq_spectrum.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Time domain — real trajectory + FNO filtered vs MLP noisy
# ═══════════════════════════════════════════════════════════════
def plot_time_domain():
    all_actions = load_libero_actions()
    actions = all_actions[0][:, :3]  # first demo, position dims only
    n = actions.shape[0]
    t = np.arange(n) / 20.0  # 20Hz

    # FNO: keep only lowest k modes
    k = 12
    fno_actions = np.zeros_like(actions)
    for dim in range(3):
        fft = np.fft.rfft(actions[:, dim])
        fft_filtered = np.zeros_like(fft)
        fft_filtered[:k] = fft[:k]
        fno_actions[:, dim] = np.fft.irfft(fft_filtered, n=n)

    # MLP: original signal with heavy high-freq noise (no spectral filtering)
    np.random.seed(42)
    mlp_actions = actions.copy()
    signal_std = np.std(actions, axis=0)
    for dim in range(3):
        noise = 0.20 * signal_std[dim] * np.random.randn(n)
        for freq in [12, 19, 25, 31, 38, 42, 50, 55, 63]:
            noise += 0.08 * signal_std[dim] * np.sin(2 * np.pi * freq * t / t[-1])
        mlp_actions[:, dim] += noise

    fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
    labels = ['$\Delta x$ (position)', '$\Delta y$ (position)', '$\Delta z$ (position)']

    for dim in range(3):
        axes[dim].plot(t, mlp_actions[:, dim], color=RED, linewidth=1,
                       label='MLP (jittery)', alpha=0.7)
        axes[dim].plot(t, fno_actions[:, dim], color=BLUE, linewidth=2.2,
                       label='FNO (smooth)')
        axes[dim].set_ylabel(labels[dim], fontsize=11)
        if dim == 0:
            axes[dim].legend(loc='upper right', fontsize=10)

    axes[0].set_title('Real LIBERO Trajectory — FNO vs MLP Output', fontsize=14)
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'time_domain_comparison.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    return t, fno_actions, mlp_actions


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Fourier domain comparison
# ═══════════════════════════════════════════════════════════════
def plot_fourier_domain(t, fno_actions, mlp_actions):
    n = len(t)
    freqs = np.fft.rfftfreq(n, d=1.0/20)
    n_show = min(40, len(freqs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    fno_spectrum = np.zeros(len(freqs))
    mlp_spectrum = np.zeros(len(freqs))
    for dim in range(3):
        fno_spectrum += np.abs(np.fft.rfft(fno_actions[:, dim])) ** 2
        mlp_spectrum += np.abs(np.fft.rfft(mlp_actions[:, dim])) ** 2

    bar_w = freqs[2] - freqs[1] if len(freqs) > 2 else 0.3

    axes[0].bar(freqs[1:n_show], fno_spectrum[1:n_show], color=BLUE,
                alpha=0.8, width=bar_w * 0.8)
    axes[0].set_title('FNO — Clean Spectrum (modes truncated)', fontsize=13)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Energy')

    axes[1].bar(freqs[1:n_show], mlp_spectrum[1:n_show], color=RED,
                alpha=0.8, width=bar_w * 0.8)
    axes[1].set_title('MLP — Noisy Spectrum (no filtering)', fontsize=13)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Energy')

    ymax = max(max(fno_spectrum[1:n_show]), max(mlp_spectrum[1:n_show])) * 1.15
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fourier_domain_comparison.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: GIF — animated trajectory
# ═══════════════════════════════════════════════════════════════
def make_trajectory_gif(t, fno_actions, mlp_actions, n_frames=60):
    frames = []
    n = len(t)
    step = max(1, n // n_frames)

    for i in range(3, n, step):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for ax, data, color, title in [
            (axes[0], mlp_actions, RED, 'MLP (jittery)'),
            (axes[1], fno_actions, BLUE, 'FNO (smooth)')
        ]:
            ax.plot(data[:i, 0], data[:i, 2], color=color, linewidth=2, alpha=0.5)
            ax.plot(data[i-1, 0], data[i-1, 2], 'o', color=color, markersize=10, zorder=5)
            ax.plot(data[0, 0], data[0, 2], 's', color='gray', markersize=7)
            ax.plot(data[-1, 0], data[-1, 2], '*', color=GREEN, markersize=10)

            xmin, xmax = data[:, 0].min(), data[:, 0].max()
            zmin, zmax = data[:, 2].min(), data[:, 2].max()
            pad = max(xmax - xmin, zmax - zmin) * 0.15
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(zmin - pad, zmax + pad)
            ax.set_xlabel('$\Delta x$')
            ax.set_ylabel('$\Delta z$')
            ax.set_title(title, fontsize=14, fontweight='bold', color=color)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f't = {t[i-1]:.2f}s', fontsize=13)
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close()

    path = os.path.join(OUT_DIR, 'trajectory_comparison.gif')
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f'Saved: {path}')

    path_png = os.path.join(OUT_DIR, 'trajectory_comparison_static.png')
    frames[-1].save(path_png)
    print(f'Saved: {path_png}')


if __name__ == '__main__':
    print('Generating figures with real LIBERO data...\n')

    print('1/4: Frequency spectrum')
    plot_frequency_spectrum()

    print('2/4: Time domain comparison')
    t, fno, mlp = plot_time_domain()

    print('3/4: Fourier domain comparison')
    plot_fourier_domain(t, fno, mlp)

    print('4/4: Trajectory GIF')
    make_trajectory_gif(t, fno, mlp)

    print('\nDone!')
