"""
Inspect LIBERO demonstration data: print shapes, dtypes, ranges.
Run after downloading LIBERO datasets.
"""

import sys
import os
import glob
import h5py
import numpy as np


def inspect_hdf5(path: str):
    """Print structure and shapes of an HDF5 demo file."""
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")

    with h5py.File(path, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())}")

        if 'data' in f:
            data = f['data']
            demos = list(data.keys())
            print(f"Number of demos: {len(demos)}")

            # Inspect first demo
            demo = data[demos[0]]
            print(f"\nDemo '{demos[0]}' keys: {list(demo.keys())}")

            for key in demo.keys():
                item = demo[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
                    arr = item[:]
                    print(f"    range: [{arr.min():.4f}, {arr.max():.4f}], mean={arr.mean():.4f}")
                elif isinstance(item, h5py.Group):
                    print(f"  {key}/ (group):")
                    for subkey in item.keys():
                        sub = item[subkey]
                        if isinstance(sub, h5py.Dataset):
                            print(f"    {subkey}: shape={sub.shape}, dtype={sub.dtype}")
                            arr = sub[:]
                            print(f"      range: [{arr.min():.4f}, {arr.max():.4f}]")

            # Episode length
            if 'actions' in demo:
                print(f"\nEpisode length: {demo['actions'].shape[0]} steps")
                print(f"Action dim: {demo['actions'].shape[1]}")
        else:
            print("No 'data' key found. Keys:", list(f.keys()))
            # Try to explore structure
            def print_structure(g, prefix=''):
                for key in g.keys():
                    item = g[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"{prefix}{key}/")
                        print_structure(item, prefix + '  ')
            print_structure(f)


def main():
    # Search for LIBERO HDF5 files
    search_paths = [
        os.path.expanduser("~/libero/**/*.hdf5"),
        os.path.expanduser("~/data/libero/**/*.hdf5"),
        os.path.expanduser("~/.libero/**/*.hdf5"),
        "/data/libero/**/*.hdf5",
        "../data/**/*.hdf5",
    ]

    found_files = []
    for pattern in search_paths:
        found_files.extend(glob.glob(pattern, recursive=True))

    if not found_files:
        print("No LIBERO HDF5 files found. Searched:")
        for p in search_paths:
            print(f"  {p}")
        print("\nDownload LIBERO data first. See README for instructions.")
        return

    print(f"Found {len(found_files)} HDF5 files:")
    for f in found_files[:5]:
        print(f"  {f}")
    if len(found_files) > 5:
        print(f"  ... and {len(found_files)-5} more")

    # Inspect first file
    inspect_hdf5(found_files[0])


if __name__ == '__main__':
    main()
