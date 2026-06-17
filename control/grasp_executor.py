"""Gripper closure / opening and grasp-success detection."""
import numpy as np


# Robotiq85 in robosuite: -1 = fully open, +1 = fully closed.
_GRIPPER_OPEN   = -1.0
_GRIPPER_CLOSE  =  1.0


class GraspExecutor:
    def close_gripper(self, env, n_steps: int = 30):
        """Step env with close action for n_steps. Returns grasp-success bool."""
        eef_pos = env.get_robot_eef_pos()
        print(f"    [GraspClose] EEF (wrist) before close: xyz={eef_pos}  z={eef_pos[2]:.4f}m")

        action = np.zeros(7)
        action[-1] = _GRIPPER_CLOSE

        for _ in range(n_steps):
            env.step(action)

        eef_pos_after = env.get_robot_eef_pos()
        print(f"    [GraspClose] EEF (wrist) after close:  xyz={eef_pos_after}  z={eef_pos_after[2]:.4f}m")

        return self.check_grasp_success(env, env.get_obj_names())

    def open_gripper(self, env, n_steps: int = 20):
        action = np.zeros(7)
        action[-1] = _GRIPPER_OPEN

        for _ in range(n_steps):
            env.step(action)

    def check_grasp_success(self, env, obj_names) -> bool:
        """True iff any gripper geom is in contact with an object body/geom."""
        obj_names = set(obj_names)

        try:
            robot = env.robots[0]
            gripper_geom_names = set(g for g in robot.gripper.contact_geoms)
        except Exception:
            gripper_geom_names = set()

        gripper_geom_names.update({
            "gripper0_finger1", "gripper0_finger2",
            "robot0_leftfinger", "robot0_rightfinger",
        })

        # Body-based detection, works when item geoms are unnamed.
        obj_body_ids = set()
        for name in obj_names:
            try:
                bid = env.sim.model.body_name2id(name)
                obj_body_ids.add(bid)
            except Exception:
                pass

        _item_geoms = []
        for gid in range(env.sim.model.ngeom):
            gname = env.sim.model.geom_id2name(gid)
            if gname and ("item" in gname.lower() or "object" in gname.lower()):
                _item_geoms.append(gname)
        if _item_geoms:
            print(f"    [GraspCheck] Item geoms in sim: {_item_geoms[:6]}")
        else:
            print(f"    [GraspCheck] WARNING: no 'item'/'object' geoms found by name "
                  f"(ngeom={env.sim.model.ngeom}). Using body-ID detection.")

        # Geom-name fallback for named geoms.
        obj_geom_names = set()
        for name in obj_names:
            obj_geom_names.add(f"{name}_g0")
            obj_geom_names.add(name)

        try:
            gripper_contacts = []
            for i in range(env.sim.data.ncon):
                contact = env.sim.data.contact[i]
                gid1, gid2 = contact.geom1, contact.geom2
                g1 = env.sim.model.geom_id2name(gid1) or ""
                g2 = env.sim.model.geom_id2name(gid2) or ""
                b1 = env.sim.model.geom_bodyid[gid1]
                b2 = env.sim.model.geom_bodyid[gid2]
                b1_name = env.sim.model.body_id2name(b1) or ""
                b2_name = env.sim.model.body_id2name(b2) or ""

                g1_is_gripper = g1 in gripper_geom_names
                g2_is_gripper = g2 in gripper_geom_names

                b1_is_obj = b1 in obj_body_ids or b1_name in obj_names
                b2_is_obj = b2 in obj_body_ids or b2_name in obj_names
                g1_is_obj = g1 in obj_geom_names
                g2_is_obj = g2 in obj_geom_names

                if g1_is_gripper or g2_is_gripper:
                    gripper_contacts.append((g1 or f"geom{gid1}", g2 or f"geom{gid2}",
                                             b1_name, b2_name))
                    if (g1_is_gripper and (b2_is_obj or g2_is_obj)) or \
                       (g2_is_gripper and (b1_is_obj or g1_is_obj)):
                        print(f"    [GraspCheck] SUCCESS: gripper touching object body "
                              f"'{b1_name}' / '{b2_name}'")
                        return True

            if gripper_contacts:
                print(f"    [GraspCheck] Gripper has {len(gripper_contacts)} contacts "
                      f"(none with target objects):")
                for g1, g2, b1, b2 in gripper_contacts[:6]:
                    print(f"      geom: {g1!r} <-> {g2!r}  body: {b1!r} <-> {b2!r}")
            else:
                print(f"    [GraspCheck] No gripper contacts at all (ncon={env.sim.data.ncon})")
        except Exception as e:
            print(f"    [GraspCheck] Exception: {e}")

        return False

    def get_gripper_width(self, env) -> float:
        try:
            return env.robots[0].gripper.current_action
        except Exception:
            return 0.0
