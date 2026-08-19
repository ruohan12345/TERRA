import os


def _first_existing_path(candidates, kind, description):
    predicate = os.path.isdir if kind == "directory" else os.path.isfile
    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(candidate))
        if path in checked:
            continue
        checked.append(path)
        if predicate(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {description}. Checked: " + ", ".join(checked)
    )


def resolve_glorys_sequential_root(
    norm_type,
    data_precision="fp16",
    explicit_root=None,
):
    norm_type = str(norm_type).strip().lower()
    data_precision = str(data_precision).strip().lower()
    if norm_type not in ("mm", "zs"):
        raise ValueError(f"Unsupported GLORYS norm_type={norm_type!r}")
    if data_precision not in ("fp16", "bf16"):
        raise ValueError(f"Unsupported GLORYS data_precision={data_precision!r}")

    norm_override = os.environ.get(
        "TERRA_GLORYS_MM_ROOT" if norm_type == "mm" else "TERRA_GLORYS_ZS_ROOT"
    )
    return _first_existing_path(
        [explicit_root, os.environ.get("TERRA_GLORYS_SEQUENTIAL_ROOT"), norm_override],
        "directory",
        f"GLORYS {norm_type}/{data_precision} sequential dataset",
    )


def resolve_glorys_parallel_root(explicit_root=None):
    return _first_existing_path(
        [explicit_root, os.environ.get("TERRA_GLORYS_PARALLEL_ROOT")],
        "directory",
        "prepared GLORYS window-parallel dataset",
    )


def get_glorys_parallel_file_path_prefix(
    embedding_parallel_type,
    height,
    width,
    wp_topo,
    patch_size,
    window_size,
    padding_scale,
    norm_type="mm",
    data_precision="fp16",
    padding_spec=None,
):
    del (
        embedding_parallel_type,
        height,
        width,
        wp_topo,
        patch_size,
        window_size,
        padding_scale,
        norm_type,
        data_precision,
    )
    explicit_root = None
    if isinstance(padding_spec, dict):
        explicit_root = padding_spec.get("glorys_parallel_root")
    return resolve_glorys_parallel_root(explicit_root)


def resolve_glorys_mask_path(explicit_path=None):
    return _first_existing_path(
        [explicit_path, os.environ.get("TERRA_GLORYS_MASK_PATH")],
        "file",
        "GLORYS land-sea mask",
    )
