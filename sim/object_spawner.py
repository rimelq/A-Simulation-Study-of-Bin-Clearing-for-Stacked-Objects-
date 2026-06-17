"""ObjectSpawner: reads poses CSV, generates per-item MuJoCo XML, and applies container transform."""
import os
import csv
import random
import numpy as np
from scipy.spatial.transform import Rotation


# default mesh package, resolved relative to repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
STL_DIR   = os.path.join(_PACKAGE_ROOT, "assets", "square_22_pkg", "stl_vis")
POSES_CSV = os.path.join(_PACKAGE_ROOT, "assets", "square_22_pkg",
                         "poses_relative_to_container.csv")

_XML_TEMPLATE = """\
<mujoco model="{name}">
  <asset>
    <mesh name="{name}_mesh" file="{abs_stl_path}" scale="0.25 0.25 0.25"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom type="mesh" mesh="{name}_mesh"
              rgba="0.65 0.50 0.35 1"
              solimp="0.998 0.998 0.001" solref="0.001 1"
              density="500"
              friction="0.9 0.005 0.0001"
              condim="4"
              contype="1" conaffinity="1"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.02" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0  0.02" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0.02 0.02 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""


def _euler_to_quat_wxyz(roll, pitch, yaw):
    r = Rotation.from_euler("xyz", [roll, pitch, yaw])
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


class ObjectSpawner:
    """Reads the poses CSV, generates per-object MuJoCo XML, and provides object lists."""

    def __init__(
        self,
        stl_dir: str = STL_DIR,
        poses_csv: str = POSES_CSV,
        generated_xml_dir: str = None,
        scale: float = 0.25,
        n_objects: int = None,
        seed: int = 42,
    ):
        self.stl_dir = stl_dir
        self.poses_csv = poses_csv
        self.scale = scale
        self.n_objects = n_objects
        self.seed = seed

        if generated_xml_dir is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            generated_xml_dir = os.path.join(_here, "..", "assets", "generated_xml")
        self.generated_xml_dir = os.path.abspath(generated_xml_dir)
        os.makedirs(self.generated_xml_dir, exist_ok=True)

        self._all_rows = self._load_csv()
        self._selected = self._select_subset()

    def _load_csv(self):
        rows = []
        with open(self.poses_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kind = row.get("kind", "").strip()
                if kind == "item":
                    rows.append(row)
        return rows

    def _select_subset(self):
        """Pick the n_objects items with the lowest z (the real bottom layer of the stack).

        Pure z-sort: an earlier version quantised z into 0.01 buckets and tie-broke by XY
        distance to the bin centre, but the whole bottom layer falls into ONE bucket, so the
        centre-distance tiebreaker dominated and picked overlapping items. Sorting by raw z
        returns the true bottom layer, spread cleanly across the bin floor.
        """
        rows = list(self._all_rows)
        if self.n_objects is not None and self.n_objects < len(rows):
            rows.sort(key=lambda r: float(r["z"]))
            rows = rows[:self.n_objects]
        return rows

    def _row_to_dict(self, row):
        name = row["name"].strip()
        mesh_file = row["mesh"].strip()
        # CSV lists .glb but on-disk files are .stl
        stl_filename = os.path.splitext(mesh_file)[0] + ".stl"
        abs_stl = os.path.join(self.stl_dir, stl_filename)

        x = float(row["x"]); y = float(row["y"]); z = float(row["z"])
        roll = float(row["roll"]); pitch = float(row["pitch"]); yaw = float(row["yaw"])

        # CSV positions are in unscaled coords. apply same mesh scale
        rel_pos = np.array([x, y, z], dtype=np.float64) * self.scale
        rel_quat = _euler_to_quat_wxyz(roll, pitch, yaw)

        xml_path = os.path.join(self.generated_xml_dir, f"{name}.xml")

        return {
            "name": name,
            "abs_stl": abs_stl,
            "xml_path": xml_path,
            "rel_pos": rel_pos,
            "rel_quat": rel_quat,
        }

    def generate_xml_files(self):
        """Generate MuJoCo XML for all selected objects. idempotent overwrite.
        STL path is written relative to the XML file location so the resulting
        XML is install-agnostic."""
        for row in self._selected:
            obj = self._row_to_dict(row)
            rel_stl = os.path.relpath(
                obj["abs_stl"],
                start=os.path.dirname(obj["xml_path"]),
            )
            xml_content = _XML_TEMPLATE.format(
                name=obj["name"],
                abs_stl_path=rel_stl,
            )
            with open(obj["xml_path"], "w") as f:
                f.write(xml_content)

    def get_object_list(self):
        """Returns list of dicts: name, xml_path, rel_pos, rel_quat (wxyz)."""
        return [self._row_to_dict(row) for row in self._selected]

    def get_world_positions(self, container_world_pos, container_world_quat):
        """Apply container rigid-body transform to each object's relative pose.

        container_world_quat is (w,x,y,z). Returns list of dicts: name, world_pos, world_quat.
        """
        cq = container_world_quat  # wxyz
        r_container = Rotation.from_quat([cq[1], cq[2], cq[3], cq[0]])  # scipy xyzw

        # Blender->GLTF Z-up to Y-up axis swap: a disk flat in Blender (face XY) comes
        # out face-XZ in the STL. +90deg around X undoes this, so the disk lies flat
        # in MuJoCo (+Z up) and reads as a circle from above.
        r_gltf_correction = Rotation.from_euler("x", 90, degrees=True)

        result = []
        for row in self._selected:
            obj = self._row_to_dict(row)
            world_pos = container_world_pos + r_container.apply(obj["rel_pos"])
            rq = obj["rel_quat"]  # wxyz
            r_rel = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]])
            r_world = r_container * r_gltf_correction * r_rel
            xyzw = r_world.as_quat()
            world_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            result.append({
                "name": obj["name"],
                "world_pos": world_pos,
                "world_quat": world_quat,
            })
        return result

    def resample(self, n_objects=None, seed=None):
        """Re-select a new subset (call before generate_xml_files)."""
        if n_objects is not None:
            self.n_objects = n_objects
        if seed is not None:
            self.seed = seed
        self._selected = self._select_subset()
