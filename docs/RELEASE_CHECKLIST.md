# Release Checklist

- [ ] Choose and add a license file.
- [ ] Replace placeholder checkpoint paths in `README.md` and `configs/*.yaml`.
- [ ] Upload or link public pruned checkpoints separately from GitHub source code.
- [ ] Decide whether to publish small example data only or also host full rewritten/filtering datasets externally.
- [ ] Confirm `OPENAI_API_KEY` and `HF_TOKEN` are not committed.
- [ ] Run a data-pipeline smoke test with `--limit 2` and confirm divergence scores are written.
- [ ] Confirm `persona_alpaca_specific_user_*.json` questions are excluded from train scoring/selection.
- [ ] Add or link keep-index tensors separately from GitHub source if releasing pruned checkpoints.
- [ ] Run a smoke test with one small trait file.
- [ ] Run the target deterministic eval for all users and save `aggregate_summary.json`.
- [ ] Add the final paper citation once the proceedings metadata is available.
