import os
import mujoco
import time
from mujoco import viewer  # Add this import to fix the AttributeError

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

xml_paths = [
    os.path.join(project_dir, "data/assets/mjcf/smpl/6803e1fa_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/aaab922b_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/0a1ece18_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/638a4fb7_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/0a1ece18_smpl.xml"),
    os.path.join(project_dir, "data/assets/mjcf/smpl/75e01b05_smpl.xml"),
]


def simulate_model(xml_path):
    print(f"\nLoading and simulating model: {xml_path}")
    
    # Load the model from XML
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Set initial root z position
    mj_data.qpos[2] = 1.0
    
    # Run forward kinematics initially
    mujoco.mj_forward(mj_model, mj_data)
    
    # Launch viewer and simulate
    with viewer.launch_passive(mj_model, mj_data) as v:  # Use viewer instead of mujoco.viewer
        start = time.time()
        step_count = 0
        while v.is_running():
            step_start = time.time()
            
            # Simulate one step
            mujoco.mj_step(mj_model, mj_data)
            step_count += 1
            
            # Print diagnostics every 10 steps
            if step_count % 10 == 0:
                max_vel = max(abs(vel) for vel in mj_data.qvel) if len(mj_data.qvel) > 0 else 0
                max_accel = max(abs(acc) for acc in mj_data.qacc) if len(mj_data.qacc) > 0 else 0
                print(f"Time: {mj_data.time:.2f}s, Steps: {step_count}, Max velocity: {max_vel:.4f}, Max acceleration: {max_accel:.4f}, Contacts: {mj_data.ncon}")
                
                # Check for instability (e.g., if max_vel > some threshold, like 50)
                if max_vel > 50:
                    print("WARNING: Potential instability detected (high velocity)")
            
            # Toggle contact points visualization
            with v.lock():
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mj_data.time % 2)
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1  # Show contact forces
            
            # Sync viewer
            v.sync()
            
            # Maintain simulation timestep
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if step_count >= 1000:
                break

if __name__ == "__main__":
    for path in xml_paths:
        simulate_model(path)
        print(f"Simulation for {path} ended. Press Enter to continue to next model...")
        input()  # Wait for user input to proceed to next model