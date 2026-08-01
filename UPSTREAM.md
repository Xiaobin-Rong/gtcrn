# Upstream attribution and repository policy

This repository is a fork of the official GTCRN implementation.

- Upstream: <https://github.com/Xiaobin-Rong/gtcrn>
- Original author: Rong Xiaobin
- Upstream license: MIT; see `LICENSE`
- Paper: *GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources*, ICASSP 2024

The `main` branch should remain easy to compare and synchronize with upstream.
MIC4CODE development is performed on topic branches and integrated through
pull requests.

## Weight policy

The official public checkpoints and ONNX files already distributed by the
upstream repository may remain in Git history. Newly trained or fine-tuned
weights are private artifacts and must not be committed to this public fork.

Private artifacts include PyTorch checkpoints, exported fine-tuned ONNX files,
optimizer states, experiment directories, training logs that contain private
paths, and any training data. Store them in an access-controlled model registry
or object store. Public commits may contain configurations, code, aggregate
metrics, model version identifiers, and SHA-256 checksums.

Before every push, review both commands:

```bash
git status --short
git diff --cached --name-only
```

If a private weight is ever committed, deleting it in a later commit is not
sufficient because it remains in Git history and may already exist in forks.
