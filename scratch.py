import zarr
import numpy as np
import os

def check_zarr(path):
    print(f"Checking {path}")
    store = zarr.open_group(path, mode='r')
    
    keys = list(store.keys())
    
    if 'right.obs_ee_pose' in keys:
        obs_key = 'right.obs_ee_pose'
    else:
        print(f"No obs key found. Keys: {keys}")
        return
        
    if 'right.action_ee_pose' in keys:
        action_key = 'right.action_ee_pose'
    elif 'right.cmd_ee_pose' in keys:
        action_key = 'right.cmd_ee_pose'
    else:
        print(f"No action key found. Keys: {keys}")
        return
        
    obs = np.array(store[obs_key])
    action = np.array(store[action_key])
    
    print(f"obs shape: {obs.shape}, action shape: {action.shape}")
    
    if obs.shape[0] < 2:
        print("Too short")
        return
        
    # Check if action is next obs
    diff = np.abs(action[:-1] - obs[1:len(action)])
    print(f"Mean abs diff between action[t] and obs[t+1]: {np.mean(diff)}")
    print(f"Max abs diff: {np.max(diff)}")
    
    # Also check if action is current obs
    diff_curr = np.abs(action - obs[:len(action)])
    print(f"Mean abs diff between action[t] and obs[t]: {np.mean(diff_curr)}")

human_zarr = "/data/madhavan/pick_red_mug_human/egoverse_human_mixes_15hz/mix1_old200_new200/train/old_s0_e000000_2026-07-15-22-40-15-000000.zarr"
robot_zarr = "/data/madhavan/pick_red_mug_franka_egoverse_left_15hz/episode_000000.zarr"

check_zarr(human_zarr)
print("----------------")
check_zarr(robot_zarr)
