import os
import mujoco

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

xml_paths = [
    os.path.join(project_dir, "data/assets/mjcf/smpl/6803e1fa_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/aaab922b_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/0a1ece18_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/638a4fb7_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/0a1ece18_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/75e01b05_smpl.xml"),
]

for xml_path in xml_paths:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[2] = 1.2  # your override
    mujoco.mj_forward(model, data)
    
    print(f"\nModel: {xml_path}")
    print(f"Number of contacts: {data.ncon}")
    if data.ncon > 0:
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_name = model.geom(contact.geom1).name if contact.geom1 >= 0 else "None"
            geom2_name = model.geom(contact.geom2).name if contact.geom2 >= 0 else "None"  # one might be floor ("floor")
            dist = contact.dist
            print(f"Contact {i}: Geoms '{geom1_name}' and '{geom2_name}', penetration depth: {dist:.4f}")
    else:
        print("No initial contacts (feet may be above ground).")