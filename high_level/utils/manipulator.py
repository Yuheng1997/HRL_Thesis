import numpy as np
import torch
import torchgeometry as tgm
import xml.etree.ElementTree as ET


class LinkTF:
    def __init__(self, name, rpy, xyz, mass, inertia):
        self.name = name
        self.rpy = rpy
        self.xyz = np.array(xyz, dtype=np.float32)
        self.mass = mass
        self.inertia = inertia


class JointTF:
    def __init__(self, jtype, parent, child, rpy, xyz, axis, lb, ub):
        self.parent = parent
        self.child = child
        self.rpy = torch.tensor(rpy, dtype=torch.float32)
        self.roll = rpy[0]
        self.pitch = rpy[1]
        self.yaw = rpy[2]
        self.xyz = torch.tensor(xyz, dtype=torch.float32)
        self.axis = torch.tensor(axis) if axis is not None else None
        self.fixed = jtype == "fixed"
        self.lb = lb
        self.ub = ub
        self.Rb = tgm.angle_axis_to_rotation_matrix(self.rpy.unsqueeze(0))  # TODO: Check if tfg.rotation_matrix_3d.from_euler(self.rpy) is met

    def Rq(self, q):
        if self.fixed:
            R = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(q.shape[:-1] + (1, 1))
            return R
        else:
            axis_expanded = self.axis.unsqueeze(-1).unsqueeze(-1)
            q_expanded = q[..., None] * axis_expanded
            return tgm.angle_axis_to_rotation_matrix(q_expanded)  # TODO: Check if tfg.rotation_matrix_3d.from_euler(self.rpy) is met

    def R(self, q):
        if self.axis is None:
            return self.Rb
        Rq = self.Rq(q)
        return torch.matmul(self.Rb, Rq)

    def T(self, q):
        R = self.R(q)
        Rp = torch.cat([R, self.xyz[:, None]], dim=-1)
        T = torch.cat([Rp, torch.tensor([0., 0., 0., 1.])[None]], dim=0)
        return T


class Iiwa:
    def __init__(self, urdf_path):
        self.joints, self.links = Iiwa.parse_urdf(urdf_path)
        self.n_dof = len(self.joints)

    @staticmethod
    def parse_urdf(urdf_path):
        root = ET.parse(urdf_path).getroot()
        joints = []
        for joint in root.findall("joint"):
            jtype = joint.get('type')
            parent = joint.find("parent").get('link')
            child = joint.find("child").get('link')
            lb = float(joint.find("limit").get("lower")) if joint.find("limit") is not None else 0.0
            ub = float(joint.find("limit").get("upper")) if joint.find("limit") is not None else 0.0
            rpy = [float(x) for x in joint.find("origin").get('rpy').split()]
            xyz = [float(x) for x in joint.find("origin").get('xyz').split()]
            axis = joint.find("axis")
            if axis is not None:
                axis = [float(x) for x in axis.get('xyz').split()]
            joints.append(JointTF(jtype, parent, child, rpy, xyz, axis, lb, ub))
            # end at striker_tip
            if joints[-1].child.endswith("striker_tip"):
                break
        links = []
        for link in root.findall("link"):
            name = link.get("name")
            inertial = link.find("inertial")
            if inertial is None:
                continue
            rpy = [float(x) for x in inertial.find("origin").get('rpy').split()]
            xyz = [float(x) for x in inertial.find("origin").get('xyz').split()]
            mass = float(inertial.find("mass").get("value"))
            I = {k: float(v) for k, v in inertial.find("inertia").items()}
            inertia = torch.tensor(
                [[I["ixx"], I["ixy"], I["ixz"]], [I["ixy"], I["iyy"], I["iyz"]], [I["ixz"], I["iyz"], I["izz"]]])
            links.append(LinkTF(name, rpy, xyz, mass, inertia))
        return joints, links
