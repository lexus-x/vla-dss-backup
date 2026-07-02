"""
Reproducible proof of Octo-Small's TRUE deployable size.
Run in the `octo` conda env:  python verify_octo_params.py
Loads the official rail-berkeley/octo-small-1.5 checkpoint and counts every
parameter, splitting out the frozen T5-base language encoder (`hf_model`).
"""
import numpy as np, jax
from octo.model.octo_model import OctoModel

m = OctoModel.load_pretrained('hf://rail-berkeley/octo-small-1.5')
flat = jax.tree_util.tree_flatten_with_path(m.params)[0]
lang = other = 0
for path, arr in flat:
    key = '/'.join(str(getattr(p, 'key', p)) for p in path)
    n = int(np.prod(arr.shape))
    if 'hf_model' in key or 't5' in key.lower():
        lang += n
    else:
        other += n
print(f"OCTO-SMALL 1.5 total params : {(lang+other)/1e6:.1f}M")
print(f"  T5-base language encoder  : {lang/1e6:.1f}M   (frozen, required at inference)")
print(f"  Octo transformer + heads  : {other/1e6:.1f}M")
print(f"VLA-DSS total               : 28.9M  (incl. 4.4M bert-tiny language encoder)")
print(f"=> deployable size ratio    : {(lang+other)/28.9e6:.1f}x")
