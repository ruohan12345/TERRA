# Tests

- `unit/`: deterministic CPU tests for assignment and configuration semantics.
- `distributed/correctness/`: serial-versus-parallel gradient and loss checks.

Run unit tests with `pytest -q tests/unit`. Distributed tests require the GPU count documented in their README.
