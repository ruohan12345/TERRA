# Public release checklist

- [x] Select and add the repository license.
- [x] Replace placeholder author and repository metadata in `CITATION.cff`.
- [x] Record the CREDIT source revision, Apache-2.0 license, and adaptation notice.
- [x] Record the Swin Transformer source and include its MIT license.
- [ ] Verify the precise PhysicsNeMo provenance of `credit/domain_parallel` and preserve any required upstream headers.
- [x] Confirm that no private paths, IP addresses, usernames, or credentials remain.
- [ ] Run installation in a fresh Python 3.10 environment.
- [ ] Run CPU/unit tests and 1-GPU fake-input smoke test.
- [ ] Run 2/8-GPU serial-versus-parallel correctness tests.
- [ ] Verify `terra_m1_ragged_auto` contiguous ownership and topology/mode validation.
- [ ] Verify `source test_env.sh && bash test_g.sh` on one eight-GPU node.
- [ ] Publish optional raw logs, checkpoints, and timelines separately with SHA-256 checksums.
