from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_grid_root() -> Path:
    return repository_root().parent / "GRiD"


def patch_fixed_target_serial_chain_calls(generated: Path, fixed_targets: str):
    """Patch GRiD fixed-target helper calls for serial-chain generated headers."""

    if not fixed_targets or not generated.exists():
        return

    text = generated.read_text()
    patched = text
    for target in [name.strip() for name in fixed_targets.split(",") if name.strip()]:
        patched = patched.replace(
            f"end_effector_pose_inner_{target}<T>(s_eePos, s_q, s_XmatsHom, s_temp);",
            f"end_effector_pose_inner_{target}<T>(s_eePos, s_q, s_XmatsHom, s_topology_helpers, s_temp);",
        )
        patched = patched.replace(
            f"end_effector_pose_gradient_inner_{target}<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);",
            f"end_effector_pose_gradient_inner_{target}<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_topology_helpers, s_temp);",
        )
    if patched != text:
        generated.write_text(patched)


def patch_floating_forward_dynamics_input_copy(generated: Path):
    """Patch GRiD floating FD batched kernel to copy all q/qd/u entries."""

    if not generated.exists():
        return

    text = generated.read_text()
    patched = text.replace(
        "for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){\n"
        "                s_q_qd_u[ind] = d_q_qd_u_k[ind];\n"
        "            }\n"
        "            __syncthreads();\n"
        "            // compute\n"
        "            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);\n"
        "            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity);",
        "for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 37; ind += blockDim.x*blockDim.y){\n"
        "                s_q_qd_u[ind] = d_q_qd_u_k[ind];\n"
        "            }\n"
        "            __syncthreads();\n"
        "            // compute\n"
        "            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);\n"
        "            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity);",
    )
    if patched != text:
        generated.write_text(patched)


def run_codegen(
    *,
    urdf_path: Path,
    out_dir: Path,
    robot_name: str,
    grid_root: Path,
    floating_base: bool,
    namespace: str,
    fixed_targets: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(grid_root / "generateGRiD.py"),
        str(urdf_path),
        "-n",
        namespace,
    ]
    if floating_base:
        cmd.append("-f")
    if fixed_targets:
        cmd.extend(["-t", fixed_targets])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(grid_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=out_dir, env=env, check=True)

    generated = out_dir / f"{namespace}.cuh"
    legacy_generated = out_dir / "grid.cuh"
    if legacy_generated.exists() and generated != legacy_generated:
        shutil.copy2(legacy_generated, generated)
    if not generated.exists() and not legacy_generated.exists():
        raise RuntimeError(
            "GRiD code generation completed without producing grid.cuh. "
            "Check that the URDF is accepted by GRiD's parser."
        )
    output_header = generated if generated.exists() else legacy_generated
    patch_fixed_target_serial_chain_calls(output_header, fixed_targets)
    if floating_base:
        patch_floating_forward_dynamics_input_copy(output_header)

    metadata = {
        "robot_name": robot_name,
        "urdf_path": str(urdf_path),
        "grid_root": str(grid_root),
        "floating_base": floating_base,
        "namespace": namespace,
        "fixed_targets": fixed_targets,
        "generated_header": str(output_header),
    }
    (out_dir / "grid_codegen_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate robot-specific GRiD CUDA code.")
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--grid-root", type=Path, default=default_grid_root())
    parser.add_argument("--namespace", default="grid")
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--fixed-targets", default="")
    args = parser.parse_args(argv)

    metadata = run_codegen(
        urdf_path=args.urdf.resolve(),
        out_dir=args.out_dir.resolve(),
        robot_name=args.robot_name,
        grid_root=args.grid_root.resolve(),
        floating_base=args.floating_base,
        namespace=args.namespace,
        fixed_targets=args.fixed_targets,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
