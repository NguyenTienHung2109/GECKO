# Run the GECKO rebuttal checks on one RTX 3090

This checkout contains the launcher, frozen experiment configurations, job plan,
and input checksums for P1, the LC directed-decoder ablation, and the
matched-compute GEM–MOTION diagnostic. S3 uses its separate campaign.

## Stage inputs

Clone the `rebuttal` branch into a dedicated directory on the server's SSD.
Transfer the files listed in
`outputs/rebuttal_20261108_ready_v1/provenance/transfer_3090.files` from the
prepared workspace, preserving paths and hard links. The list includes the
18 streams, 12 reusable GEM results and their verification records, the source
snapshot, and the small qpth dependency bundle. Files already present in this
checkout can be reused if their checksums match.

From the checkout root, verify the combined tree.

```bash
campaign=outputs/rebuttal_20261108_ready_v1
sha256sum -c --quiet "$campaign/provenance/transfer_3090.sha256"
```

Use a compatible Python environment with the versions in the transferred
`outputs/e4_20260929/provenance/environment.json`. Verify that PyTorch 2.7.1+cu128
and DGL 2.5 can load on the target host.

## Launch

Choose the UUID of one idle RTX 3090. Run the launcher inside a cgroup with a
total CPU quota of eight logical CPUs and a 64 GiB memory limit. For example,
on a systemd host with user controller delegation, set `py` to the absolute
path to the Python interpreter and `gpu_uuid` to the GPU UUID, then run:

```bash
systemd-run --user --unit=gecko-rebuttal-3090 \
  --property=CPUQuota=800% --property=MemoryMax=64G \
  /bin/bash "$PWD/$campaign/scripts/queue_3090.sh" "$py" "$gpu_uuid"
```

The launcher checks both resource limits, the selected GPU, and runtime
versions. It runs the matched-compute diagnostic, six LC decoder pairs, then
36 P1 jobs. Monitor `$campaign/logs/queue_3090.log` and the per-job P1 logs.
If the unit exits before its log opens, read
`journalctl --user -u gecko-rebuttal-3090` for the launch error.

Results are written below `$campaign/runs`, with P1 summaries below
`$campaign/tables`. Check the matched-compute result for the realized time
difference and match status before reporting the comparison.
