
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # 0.9 causes too much lag. 
from datetime import datetime
import functools
# Math
import jax.numpy as jnp
import numpy as np
import jax
from jax import config  # Analytical gradients work much better with double precision.
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'high')

from dyn_visualization import vis_dynamics
from mppi import MPPI
import pdb
import matplotlib.pyplot as plt
import time
import sys
import signal, sys, matplotlib.pyplot as plt
import copy

## for signal.SIGINT
# Interrupt from keyboard (CTRL + C).
def handle_sigint(sig, frame):
    print("CtrlC detected — shutting down.")
    plt.close('all')
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# @@ use jax.generator for reproducibility
seed = 223 #113   use: 223, 7 obs in b/w , 1 : free
num_obs = 50
robot_r = 0.2
cell_size = 2*robot_r 
dim_st = 2
dim_ctrl = 2
obs_r = (cell_size)/2.0 + 0.01
pose_lim = jnp.array([[-4,-4],[4,4]])
noise_std_dev = 0.05
noise_max_limit = 2*noise_std_dev
ctrl_limit =  ((2**0.5)*robot_r)  - ( noise_max_limit )  # maximum control limit is 2^{0.5}*robot_r without the noise , with noise its robot_r - noise_max_limit

def scale_rnd_num(lim,rnd_num):
    # squashing the sample in [-1,1]
    rnd_num = jnp.tanh(rnd_num/0.5)
    # transforming the samples to [0,1]
    rnd_num = (rnd_num+1.0)/2.0
    # transfroming the samples to the actual bound 
    un_num =  lim[0,0:][...,jnp.newaxis] + (lim[1,0:][...,jnp.newaxis] - lim[0,0:][...,jnp.newaxis] )*(rnd_num)
    return un_num

def get_start_goal(pose_lim, key)->tuple:
    
    start_key,goal_key  = jax.random.split(key, 2)

    start_sampled = jax.random.normal(start_key, shape=(2,1))
    goal_sampled = jax.random.normal(goal_key, shape=(2,1))
    start = jnp.array([2.8,-2.8]).reshape((2,1)) #scale_rnd_num(pose_lim,start_sampled)
    goal =  jnp.array([-0.8,0.8]).reshape((2,1))  #scale_rnd_num(pose_lim,goal_sampled)
    return (start,goal)


def get_obs_pose( pose_lim,key,num_obs):
    #obs_array = jax.random.normal(key, shape=(num_obs,2,1))
    #obs_array = scale_rnd_num(pose_lim,obs_array)
    xmin, ymin = pose_lim[0]
    xmax, ymax = pose_lim[1]

    # Discrete grid coordinates
    x_vals = jnp.arange(xmin, xmax,step= cell_size)
    y_vals = jnp.arange(ymin, ymax,step= cell_size)
    xx, yy = jnp.meshgrid(x_vals, y_vals, indexing='ij')  # shape: (10, 10)
    grid_points = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)  # shape: (100, 2)
    # Sample num_obs unique indices from the flattened grid
    sample_indices = jax.random.choice(key, grid_points.shape[0], shape=(num_obs,), replace=False)
    # Gather the sampled points
    obs_array = grid_points[sample_indices]  # shape: (10, 2)
    return obs_array


if __name__ == "__main__":
    key = jax.random.PRNGKey(seed)
    pose_rng,global_key = jax.random.split(key, 2)
    obs_key,global_key = jax.random.split(global_key,2)
    start,goal = get_start_goal(pose_lim=pose_lim, key=pose_rng)
    obs_array:jnp.array = get_obs_pose(pose_lim = pose_lim,key=obs_key,num_obs = num_obs)
    plot_key,global_key = jax.random.split(global_key,2)
    VisualizerObj = vis_dynamics(pose_lim,start,goal,obs_array,robot_r,obs_r,noise_std_dev,noise_max_limit,plot_key)
    plt.ion()
    VisualizerObj.fig.show()
    mppi_key,global_key = jax.random.split(global_key,2)
    MppiObj  = MPPI(pose_lim,start,goal,obs_array,robot_r,obs_r,noise_std_dev,noise_max_limit,dim_st,dim_ctrl,ctrl_limit,mppi_key)
    st = copy.deepcopy(start)
    while(np.linalg.norm(st-goal)>0.1):
        optimal_control, X_optimal_seq,X_rollout= MppiObj.compute_control(st)
        key,curr_key = jax.random.split(key, 2)
        st = VisualizerObj.step(optimal_control, X_optimal_seq,X_rollout )
        #pdb.set_trace()
        #pdb.set_trace()
        #print(ctrl)
    pdb.set_trace()
        
    