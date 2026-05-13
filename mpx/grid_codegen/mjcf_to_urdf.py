from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _vec(text, default):
    if text is None:
        return list(default)
    return [float(x) for x in text.split()]


def _quat_to_matrix(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_to_rpy(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def _fmt(values):
    return " ".join(f"{x:.12g}" for x in values)


def _body_joint(body):
    joints = [j for j in body.findall("joint") if j.get("type") != "free"]
    return joints[0] if joints else None


def _add_link(robot, body):
    link = ET.SubElement(robot, "link", {"name": body.get("name")})
    inertial = body.find("inertial")
    if inertial is None:
        ET.SubElement(link, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        inertial_el = ET.SubElement(link, "inertial")
        ET.SubElement(inertial_el, "mass", {"value": "0"})
        ET.SubElement(
            inertial_el,
            "inertia",
            {"ixx": "0", "ixy": "0", "ixz": "0", "iyy": "0", "iyz": "0", "izz": "0"},
        )
        return

    com = _vec(inertial.get("pos"), [0, 0, 0])
    quat = _vec(inertial.get("quat"), [1, 0, 0, 0])
    diag = np.diag(_vec(inertial.get("diaginertia"), [0, 0, 0]))
    inertia = _quat_to_matrix(quat) @ diag @ _quat_to_matrix(quat).T
    ET.SubElement(link, "origin", {"xyz": _fmt(com), "rpy": "0 0 0"})
    inertial_el = ET.SubElement(link, "inertial")
    ET.SubElement(inertial_el, "mass", {"value": inertial.get("mass", "0")})
    ET.SubElement(
        inertial_el,
        "inertia",
        {
            "ixx": f"{inertia[0, 0]:.12g}",
            "ixy": f"{inertia[0, 1]:.12g}",
            "ixz": f"{inertia[0, 2]:.12g}",
            "iyy": f"{inertia[1, 1]:.12g}",
            "iyz": f"{inertia[1, 2]:.12g}",
            "izz": f"{inertia[2, 2]:.12g}",
        },
    )


def _add_joint(robot, parent_name, body, default_damping):
    joint = _body_joint(body)
    attrs = {"name": joint.get("name") if joint is not None else f"{parent_name}_to_{body.get('name')}"}
    attrs["type"] = "revolute" if joint is not None else "fixed"
    joint_el = ET.SubElement(robot, "joint", attrs)
    pos = _vec(body.get("pos"), [0, 0, 0])
    rpy = _quat_to_rpy(_vec(body.get("quat"), [1, 0, 0, 0]))
    ET.SubElement(joint_el, "origin", {"xyz": _fmt(pos), "rpy": _fmt(rpy)})
    ET.SubElement(joint_el, "parent", {"link": parent_name})
    ET.SubElement(joint_el, "child", {"link": body.get("name")})
    if joint is not None:
        ET.SubElement(joint_el, "axis", {"xyz": joint.get("axis", "0 0 1")})
        lower, upper = _vec(joint.get("range"), [-math.inf, math.inf])
        ET.SubElement(joint_el, "dynamics", {"damping": str(default_damping), "friction": "0"})
        ET.SubElement(
            joint_el,
            "limit",
            {
                "effort": "30",
                "lower": f"{lower:.12g}",
                "upper": f"{upper:.12g}",
                "velocity": "3.1415",
            },
        )


def _walk(robot, body, *, default_damping, parent_name=None):
    _add_link(robot, body)
    if parent_name is not None:
        _add_joint(robot, parent_name, body, default_damping)
    for child in body.findall("body"):
        _walk(robot, child, default_damping=default_damping, parent_name=body.get("name"))


def convert(mjcf_path: Path, output_path: Path, *, default_damping: float, fixed_target: str):
    mjcf = ET.parse(mjcf_path).getroot()
    worldbody = mjcf.find("worldbody")
    root_body = worldbody.find("body")
    robot = ET.Element("robot", {"name": mjcf.get("model", mjcf_path.stem)})
    _walk(robot, root_body, default_damping=default_damping)

    if fixed_target:
        target_geom = next(
            geom for geom in root_body.iter("geom") if geom.get("name") == fixed_target
        )
        parent = next(body for body in root_body.iter("body") if target_geom in list(body))
        link = ET.SubElement(robot, "link", {"name": fixed_target})
        joint = ET.SubElement(robot, "joint", {"name": fixed_target, "type": "fixed"})
        ET.SubElement(joint, "origin", {"xyz": target_geom.get("pos", "0 0 0"), "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": parent.get("name")})
        ET.SubElement(joint, "child", {"link": fixed_target})
        ET.SubElement(link, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(ET.ElementTree(robot), space="  ")
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a Z1 MuJoCo XML to a GRiD-friendly URDF.")
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--default-damping", type=float, default=0.1)
    parser.add_argument("--fixed-target", default="end_effector")
    args = parser.parse_args(argv)
    convert(
        args.mjcf.resolve(),
        args.output.resolve(),
        default_damping=args.default_damping,
        fixed_target=args.fixed_target,
    )


if __name__ == "__main__":
    main()
