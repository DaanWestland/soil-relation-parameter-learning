# Extra configurations (not used for any published result)

| File | What |
|---|---|
| `gpu_quick_lowmem.yaml` | `gpu_quick` with smaller TabICLv2 memory settings (`kv_cache: repr`, `batch_size: 4`), a fallback for GPUs that keep running out of memory despite the automatic fallbacks. Never needed for the published runs. |

Run them by path, e.g. `python -m pedopilot run configs/extra/gpu_quick_lowmem.yaml` (a config name alone
is looked up in `configs/` only). Results go to `results/<name>/`, where `<name>` is the file's `name:`.
