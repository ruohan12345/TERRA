# CREDIT-derived components

The sampling blocks and domain-parallel convolution utilities in this public
artifact are adapted from the open-source NSF NCAR Community Research Earth
Digital Intelligence Twin (CREDIT) project:

- Upstream: https://github.com/NCAR/miles-credit
- License: Apache License 2.0 (see `credit/LICENSE`)
- Sampling-block source: `credit/models/fuxi.py`
- Domain-parallel sources: `credit/domain_parallel/`
- Upstream revision reviewed for this release: `ac83d0a0d67e029af5e57babb100b7bbd0ace78e`

TERRA modifies the domain-parallel utilities to use TERRA's WP process groups,
rank manager, differentiable layout routes, distributed GroupNorm, and
checkpoint/offload instrumentation. TERRA's `CreditDownBlock` and
`CreditUpBlock` preserve the public FuXi-style sampling structure while the
checkpoint policy is applied by TERRA outside those blocks.

CREDIT and TERRA are independent projects. Use of CREDIT-derived code does not
imply endorsement by NSF NCAR or UCAR.
