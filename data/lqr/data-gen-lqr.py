#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
from tqdm import tqdm

class LQRController:
    def __init__(self, A, B, Q, R):
        """
        Initialize LQR controller for a system x(t+1) = Ax(t) + Bu(t)
        
        Args:
            A: System dynamics matrix (n x n)
            B: Control input matrix (n x m)
            Q: State cost matrix (n x n), positive definite
            R: Control cost matrix (m x m), positive definite
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._compute_lqr_gain()
        
    def _compute_lqr_gain(self):
        """Compute the optimal LQR gain matrix using the discrete algebraic Riccati equation"""
        P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        return K
    
    def get_control(self, x, x_target):
        """
        Compute control input u = -K(x - x_target)
        
        Args:
            x: Current state
            x_target: Target state
        
        Returns:
            Control input u
        """
        error = x - x_target
        u = -self.K @ error
        return u
    
    def simulate_trajectory(self, x0, x_target, n_steps=100):
        """
        Simulate a trajectory from initial state to target state
        
        Args:
            x0: Initial state
            x_target: Target state
            n_steps: Number of simulation steps
        
        Returns:
            states: Array of states (n_steps+1 x state_dim)
            controls: Array of control inputs (n_steps x control_dim)
        """
        n_state = len(x0)
        n_control = self.B.shape[1]
        
        states = np.zeros((n_steps+1, n_state))
        controls = np.zeros((n_steps, n_control))
        
        states[0] = x0
        
        for t in range(n_steps):
            x = states[t]
            u = self.get_control(x, x_target)
            controls[t] = u
            
            # Apply control constraints if needed
            # u = np.clip(u, -u_max, u_max)
            
            # Update state
            states[t+1] = self.A @ x + self.B @ u
            
        return states, controls


# In[2]:


def create_double_integrator_system():
    """
    Create a double integrator system in both X and Y dimensions
    
    Returns:
        A: System dynamics matrix
        B: Control input matrix
    """
    # Double integrator in each dimension (position and velocity for X and Y)
    # State: [x, y, vx, vy]
    # Control: [ax, ay]
    dt = 0.1  # discretization time step
    
    # Continuous-time matrices
    Ac = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    Bc = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    
    # Discretize the system (exact discretization)
    A = np.eye(4) + dt * Ac
    B = dt * Bc
    
    return A, B

def generate_trajectories(target_set, n_trajectories=100,  n_steps=30, save_dir="traj_test_set"):
    """
    Generate multiple trajectories using LQR control to stabilize random target points
    
    Args:
        n_trajectories: Number of trajectories to generate
        n_steps: Number of steps per trajectory
        save_dir: Directory to save trajectories
    """
    # Create system matrices
    A, B = create_double_integrator_system()
    
    # Create cost matrices
    n_state = A.shape[0]  # 4
    n_control = B.shape[1]  # 2
    
    # Penalize state error (higher weight on position)
    Q = np.diag([10, 10, 1, 1])
    
    # Penalize control effort
    R = np.diag([1, 1])
    
    # Initialize controller
    controller = LQRController(A, B, Q, R)
    
    # Create directory for saving trajectories
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate trajectories
    for i in tqdm(range(n_trajectories)):
        # Sample random initial state and target state
        # Initial state: position in [-5, 5], velocity in [-1, 1]
        x0 = np.concatenate([
            np.random.uniform(-5, 5, 2),  # random position
            np.random.uniform(-1, 1, 2)   # random velocity
        ])
        
        # Target state: position in [-5, 5], zero velocity
        # x_target = np.concatenate([
        #     np.random.uniform(-5, 5, 2),  # random target position
        #     np.zeros(2)                   # zero target velocity
        # ])
        target_number = np.random.choice([0,1,2,3,4])
        
        x_target = target_set[target_number,:]
        
        # Simulate trajectory
        states, controls = controller.simulate_trajectory(x0, x_target, n_steps)
        
        # Save trajectory data
        trajectory_data = {
            'states': states,
            'controls': controls,
            'initial_state': x0,
            'target_state': x_target,
            'A': A,
            'B': B,
            'Q': Q,
            'R': R
        }
        
        # Save using numpy's compressed format
        filename = os.path.join(save_dir, f"trajectory_{i:03d}.npz")
        np.savez_compressed(filename, **trajectory_data)


# In[3]:


def plot_trajectory(trajectory_file):
    """
    Plot a trajectory from a saved file
    
    Args:
        trajectory_file: Path to the saved trajectory file
    """
    # Load trajectory data
    data = np.load(trajectory_file)
    states = data['states']
    controls = data['controls']
    x0 = data['initial_state']
    x_target = data['target_state']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot trajectory in 2D space
    ax1.plot(states[:, 0], states[:, 1], 'b-')
    ax1.plot(states[0, 0], states[0, 1], 'go', label='Initial')
    ax1.plot(x_target[0], x_target[1], 'ro', label='Target')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Position Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # Plot state and control over time
    time = np.arange(len(states)) * 0.1  # assuming dt = 0.1
    
    ax2.plot(time, states[:, 0], 'r-', label='x')
    ax2.plot(time, states[:, 1], 'g-', label='y')
    ax2.plot(time, states[:, 2], 'r--', label='vx')
    ax2.plot(time, states[:, 3], 'g--', label='vy')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('State')
    ax2.set_title('State Evolution')
    ax2.grid(True)
    ax2.legend()
    
    # Create a third subplot for controls
    ax3 = fig.add_subplot(2, 2, 4)
    time_u = time[:-1]  # control has one less point
    ax3.plot(time_u, controls[:, 0], 'b-', label='u_x')
    ax3.plot(time_u, controls[:, 1], 'm-', label='u_y')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Control')
    ax3.set_title('Control Inputs')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()



# In[4]:


def main():
    """Main function to demonstrate the LQR controller and generate trajectories"""
    # Generate trajectories
    state_target_set = np.zeros((5,4))
    
    for i in range(5):
        state_target_set[i,:] = np.concatenate([np.random.uniform(-3,3,2),np.zeros(2)])
                                
    generate_trajectories(state_target_set, n_trajectories=100, n_steps=30)
    
    # Plot a sample trajectory
    plot_trajectory("traj_test_set/trajectory_000.npz")
    
    print("All trajectories generated and saved in 'trajectories' directory.")

if __name__ == "__main__":
    main()


# In[5]:


plot_trajectory("trajectories/trajectory_020.npz")


# In[6]:


state_target_set = np.zeros((5,4))

for i in range(5):
    state_target_set[i,:] = np.concatenate([np.random.uniform(-3,3,2),np.zeros(2)])
target_num = np.random.choice([0,1,2,3,4])
print(state_target_set[target_num,:])


# In[ ]:




