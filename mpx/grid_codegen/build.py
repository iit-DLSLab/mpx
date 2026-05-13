from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .generate import default_grid_root, run_codegen


def cache_key(robot_name: str, urdf_path: Path) -> str:
    return f"{robot_name}_{hashlib.sha256(urdf_path.read_bytes()).hexdigest()[:12]}"


def write_manifest(out_dir: Path, *, metadata: dict, library_name: str):
    manifest = dict(metadata)
    manifest.update(
        {
            "library_name": library_name,
            "status": "code_generated",
            "note": (
                "GRiD CUDA code was generated. The XLA FFI bridge source is intentionally "
                "kept separate from generated GRiD headers; compile a bridge exporting "
                "mpx_grid_step and mpx_grid_step_with_derivatives before setting "
                "grid_ffi_library_path."
            ),
        }
    )
    (out_dir / "grid_build_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate and stage a robot-specific GRiD backend build.")
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--grid-root", type=Path, default=default_grid_root())
    parser.add_argument("--namespace", default="grid")
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--fixed-targets", default="")
    parser.add_argument("--run-cmake", action="store_true")
    args = parser.parse_args(argv)

    urdf_path = args.urdf.resolve()
    key = cache_key(args.robot_name, urdf_path)
    out_dir = (args.out_dir or (Path.home() / ".cache" / "mpx_grid" / key)).resolve()
    metadata = run_codegen(
        urdf_path=urdf_path,
        out_dir=out_dir,
        robot_name=args.robot_name,
        grid_root=args.grid_root.resolve(),
        floating_base=args.floating_base,
        namespace=args.namespace,
        fixed_targets=args.fixed_targets,
    )
    library_name = f"libmpx_grid_{key}.so"

    template = Path(__file__).with_name("ffi_bridge_template.cu")
    if template.exists():
        shutil.copy2(template, out_dir / "ffi_bridge_template.cu")

    if args.run_cmake:
        raise NotImplementedError(
            "The GRiD XLA FFI bridge depends on the exact generated robot API and "
            "has not been compiled automatically by this helper yet."
        )

    manifest = write_manifest(out_dir, metadata=metadata, library_name=library_name)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
