import os

import xml.etree.ElementTree as ET
import numpy as np

def parse_floats(s): return np.array([float(x) for x in s.split()])

def capsule_bad(geom):
    if geom.get("type") != "capsule": 
        return False, "not capsule"
    ft = geom.get("fromto")
    sz = geom.get("size")
    if ft is None or sz is None:
        return False, "size is None"
    p = parse_floats(ft)
    a, b = p[:3], p[3:]
    r = float(sz.split()[0])
    L = np.linalg.norm(b - a)
    return (L < 1e-4) or (r > 0.49 * L), "heuristic problem"  # heuristic

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

xml_path = os.path.join(project_dir, "data/assets/mjcf/smpl/aaab922b_smpl.xml")   # your two faulty ones
xml_path = os.path.join(project_dir, "data/assets/mjcf/smpl/6803e1fa_smpl.xml")   # your two faulty ones

xml_path = os.path.join(project_dir, "data/assets/mjcf/smpl/75e01b05_smpl.xml")
xml_path = os.path.join(project_dir, "data/assets/mjcf/smpl/0e091a72_smpl.xml")


tree = ET.parse(xml_path)
root = tree.getroot()
bad = []
for g in root.findall(".//geom"):
    flag, msg = capsule_bad(g)

    if flag:
        bad.append(g.get("name"))

        print(msg)

print("Bad capsules:", bad[:50])
