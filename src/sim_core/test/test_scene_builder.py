from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim_core.frame_tree import default_robot_frame_tree
from sim_core.scene_builder import (
    build_scene_xml,
    load_physics_params,
    load_scene_geometry_params,
)


class FakeNode:
    def __init__(self, overrides: dict[str, object] | None = None) -> None:
        self.overrides = overrides or {}

    def declare_parameter(self, name: str, default_value):
        return SimpleNamespace(value=self.overrides.get(name, default_value))


def _make_meshdir(tmp_path: Path) -> Path:
    meshdir = tmp_path / "meshes"
    for relative_path in (
        "mesh_view/mesh_view.obj",
        "mesh_lidar/mesh_lidar.obj",
        "mesh_collision_env/collision.obj",
    ):
        path = meshdir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test mesh\n", encoding="ascii")
    return meshdir


def _build_default_scene(tmp_path: Path) -> ET.Element:
    node = FakeNode()
    scene_xml = build_scene_xml(
        meshdir=str(_make_meshdir(tmp_path)),
        frame_tree=default_robot_frame_tree(),
        robot_init_location=(0.0, 0.0, 0.4),
        boundary_x_min=-1.0,
        boundary_x_max=1.0,
        boundary_y_min=-1.0,
        boundary_y_max=1.0,
        scene_geometry=load_scene_geometry_params(node),
        physics=load_physics_params(node, 0.002),
        enable_left_livox=True,
        enable_right_livox=True,
    )
    return ET.fromstring(scene_xml)


def _float_values(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.attrib[attribute].split())


def test_scene_uses_configured_integrator_and_global_contact(tmp_path: Path) -> None:
    root = _build_default_scene(tmp_path)

    option = root.find("option")
    default_geom = root.find("default/geom")

    assert option is not None
    assert option.attrib["integrator"] == "implicitfast"
    assert option.attrib["timestep"] == "0.002"
    assert default_geom is not None
    assert _float_values(default_geom, "solref") == pytest.approx((0.03, 1.2))
    assert _float_values(default_geom, "solimp") == pytest.approx(
        (0.9, 0.95, 0.001, 0.5, 2.0)
    )


def test_base_dimensions_and_spherical_wheel_locations(tmp_path: Path) -> None:
    root = _build_default_scene(tmp_path)
    geoms = {
        geom.attrib["name"]: geom
        for geom in root.findall(".//geom")
        if "name" in geom.attrib
    }

    base = geoms["base_link__body"]
    assert _float_values(base, "size") == pytest.approx((0.25, 0.07))
    assert _float_values(base, "pos") == pytest.approx((0.0, 0.0, 0.15))
    assert base.attrib["condim"] == "1"

    expected_wheel_positions = {
        "base_link__wheel_front_left": (0.2, 0.2, 0.07),
        "base_link__wheel_front_right": (0.2, -0.2, 0.07),
        "base_link__wheel_rear_left": (-0.2, 0.2, 0.07),
        "base_link__wheel_rear_right": (-0.2, -0.2, 0.07),
    }
    for name, expected_position in expected_wheel_positions.items():
        wheel = geoms[name]
        assert wheel.attrib["type"] == "sphere"
        assert float(wheel.attrib["size"]) == pytest.approx(0.07)
        assert _float_values(wheel, "pos") == pytest.approx(expected_position)
        assert wheel.attrib["priority"] == "1"
        assert wheel.attrib["condim"] == "1"
        assert _float_values(wheel, "solref") == pytest.approx((0.05, 1.2))
        assert _float_values(wheel, "solimp") == pytest.approx(
            (0.8, 0.95, 0.003, 0.5, 2.0)
        )


def test_base_top_must_be_above_bottom() -> None:
    node = FakeNode(
        {
            "base_bottom_height": 0.3,
            "base_top_height": 0.2,
        }
    )

    with pytest.raises(ValueError, match="base_top_height"):
        load_scene_geometry_params(node)


def test_contact_time_constant_must_fit_timestep() -> None:
    node = FakeNode({"contact_solref": [0.003, 1.0]})

    with pytest.raises(ValueError, match=r"2 \* physics_dt"):
        load_physics_params(node, 0.002)
