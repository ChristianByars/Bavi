# Environment notes

Tested on 31 October – 1 November 2026 on one NVIDIA L40 (48 GB), Python 3.12, with a CUDA 13 driver.

## Pinned stack

- earth2studio **0.20.0** (the version published on PyPI; uv resolves it through the PyPI and NVIDIA package indexes)
- torch **2.13.0** (CUDA 13.2 wheels from the `pytorch-cu132` index)
- All other versions are resolved in `uv.lock`; use `uv sync` (not `pip install`) so you get exactly the tested set.

## Install

```bash
uv sync
uv run python -m ipykernel install --user --name bavi --display-name "bavi"
```

## Known issues and fixes

- **GPU memory between models.** Models from earlier cells stay on the GPU and can cause out-of-memory errors in later sections. Each notebook defines `free_gpu(...)` and calls it in a `finally:` block after every model, so it is released even when a cell fails. If a cell is interrupted by hand, restart the kernel.
- **`import cupy` needs pytest.** cupy imports `cupy.testing` internally, which requires `pytest`; it is listed in `dependencies` for that reason.
- **Headless runs.** `uv run jupyter nbconvert --to notebook --execute --allow-errors --inplace <notebook>` keeps the outputs of the cells that succeeded when a later cell fails.
- **Prebuilt flash-attn and torch-harmonics wheels.** flash-attn is published on PyPI only as a source distribution, and building it needs torch inside the build environment, `nvcc` and a long CUDA compile; torch-harmonics 0.8.1 is likewise source-only on PyPI and compiles CUDA extensions when it is installed. PyPI offers no prebuilt wheel of either for torch 2.13.0+cu132, so this repo expects two prebuilt cp312 wheels, `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` and `torch_harmonics-0.8.1-cp312-cp312-linux_x86_64.whl`, in a git-ignored `wheels/` folder at the repo root. `[tool.uv.sources]` points at them, and `uv sync` fails without them. Build both once against torch 2.13.0+cu132 following the upstream flash-attention and torch-harmonics build instructions, keep those exact file names, then run `uv lock`, because `uv.lock` pins the sha256 of the author's wheels. On the author's machine `wheels` is a symlink to a folder of prebuilt wheels.
- **CUDA 13 cupy and cucim.** The CUDA 13 builds (`cupy-cuda13x`, `cucim-cu13`) match torch 2.13.0+cu132. Importing earth2studio's models prints one harmless warning, `CuPy distance computation test failed with error: cuVS >= 24.12 or pylibraft < 24.12 should be installed to use this feature`; it only means the optional cuVS accelerator is not installed.
- **Git-only dependencies.** `earth2studio[sfno]` needs `makani` and `earth2studio[graphcast]` needs `weathernext`, and neither is published on PyPI. Both are listed in `dependencies` and pinned in `[tool.uv.sources]` to the same GitHub commits that earth2studio 0.20.0 pins, so `uv lock` and `uv sync` need `git` and network access to github.com.
- **Archived IFS analyses.** ECMWF's own open-data server keeps only the last four days, so the notebook reads the 2026-07-06 12 UTC analysis from ECMWF's open-data mirror on AWS (`IFS(source="aws")`; `"aws"` is earth2studio's name for that mirror). With `source="ecmwf"` the request is rejected for any older date.
