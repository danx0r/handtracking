#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco
from mujoco import viewer as mjviewer
import math
import time
import os
from math import radians, degrees
import random

class RobotArmIK:
    def __init__(self, mjcf_file):
        """
        Initialize the robot arm directly from a MuJoCo MJCF file
        
        Args:
            mjcf_file (str): Path to the MuJoCo MJCF file
        """
        # Load MJCF file directly into MuJoCo
        self.model = mujoco.MjModel.from_xml_path(mjcf_file)
        self.data = mujoco.MjData(self.model)
        
        # Find the end effector body (we'll assume it's the last body in the chain)
        # In a real application, you'd want to identify it by name
        self.end_effector_id = self.model.nbody - 1
        
        # Pre-allocate Jacobian matrices
        # jacp is the translational Jacobian (position)
        # jacr is the rotational Jacobian (orientation)
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        
    def solve_ik(self, target_pos, target_rot_matrix, max_iter=500, tol=1e-4, damping=0.1):
        """
        Solve inverse kinematics using damped least squares method
        
        Args:
            target_pos (numpy.ndarray): Target position [x, y, z] in meters
            target_rot_matrix (numpy.ndarray): Target rotation matrix (3x3)
            max_iter (int): Maximum iterations
            tol (float): Tolerance for convergence
            damping (float): Damping factor for numerical stability
            
        Returns:
            tuple: Joint angles in radians, position error in meters, orientation error in degrees, iterations
        """
        # Start with current joint configuration
        iter_count = 0
        
        while iter_count < max_iter:
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Get current end effector position
            current_pos = self.data.body(self.end_effector_id).xpos
            
            # Get current end effector orientation as rotation matrix
            current_rot = self.data.body(self.end_effector_id).xmat.reshape(3, 3)
            
            # Calculate position error
            pos_error = target_pos - current_pos
            
            # Calculate orientation error using matrix logarithm
            rel_rot = target_rot_matrix @ current_rot.T
            
            # Convert to axis-angle representation
            angle, axis = self._rotation_matrix_to_axis_angle(rel_rot)
            orient_error = axis * angle
            
            # Check for convergence
            error_norm = np.linalg.norm(pos_error) + np.linalg.norm(orient_error)
            if error_norm < tol:
                break
                
            # Get Jacobian at the end effector
            mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.end_effector_id)
            
            # Combine position and orientation error
            error = np.concatenate([pos_error, orient_error])
            
            # Combine Jacobians
            J = np.vstack([self.jacp, self.jacr])
            
            # Damped least squares method
            JTJ = J.T @ J
            damping_matrix = np.eye(JTJ.shape[0]) * damping * damping
            delta_q = np.linalg.solve(JTJ + damping_matrix, J.T @ error)
            
            # Update joint angles
            self.data.qpos[:self.model.nv] += delta_q
            
            # Check joint limits
            for i in range(min(self.model.nv, self.model.njnt)):
                if i < len(self.model.jnt_limited) and self.model.jnt_limited[i]:
                    jnt_range = self.model.jnt_range[i]
                    self.data.qpos[i] = np.clip(self.data.qpos[i], jnt_range[0], jnt_range[1])
            
            iter_count += 1
        
        # Calculate final errors
        mujoco.mj_forward(self.model, self.data)
        final_pos = self.data.body(self.end_effector_id).xpos
        final_rot = self.data.body(self.end_effector_id).xmat.reshape(3, 3)
        
        pos_error = np.linalg.norm(target_pos - final_pos)
        
        rel_rot = target_rot_matrix @ final_rot.T
        angle, _ = self._rotation_matrix_to_axis_angle(rel_rot)
        orient_error = degrees(angle)
        
        # Extract 6-DOF joint angles (assuming the first 6 are the ones we want)
        joint_angles = self.data.qpos[:6].copy()
        
        return joint_angles, pos_error, orient_error, iter_count
    
    def _rotation_matrix_to_axis_angle(self, R):
        """
        Convert rotation matrix to axis-angle representation
        
        Args:
            R (numpy.ndarray): 3x3 rotation matrix
            
        Returns:
            tuple: (angle, axis) where angle is in radians and axis is a unit vector
        """
        # Ensure R is a valid rotation matrix
        if np.linalg.det(R) < 0:
            R = -R
            
        # The angle is related to the trace
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        
        # If angle is close to zero, no clear axis of rotation
        if np.isclose(angle, 0):
            return 0, np.array([0, 0, 0])
        
        # If angle is close to pi, special case
        if np.isclose(angle, np.pi):
            # Find the eigenvalue 1 of R+I
            eigvals, eigvecs = np.linalg.eig(R + np.eye(3))
            for i in range(3):
                if np.isclose(eigvals[i], 2.0):
                    axis = eigvecs[:, i].real
                    axis = axis / np.linalg.norm(axis)
                    return angle, axis
            # Fallback if eigendecomposition fails
            axis = np.array([
                np.sqrt(max(0, (R[0, 0] + 1) / 2)),
                np.sqrt(max(0, (R[1, 1] + 1) / 2)) * np.sign(R[0, 1]),
                np.sqrt(max(0, (R[2, 2] + 1) / 2)) * np.sign(R[0, 2])
            ])
            return angle, axis
        
        # Normal case: use the skew-symmetric part of R
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        axis = axis / (2 * np.sin(angle))
        
        return angle, axis


class WorkspaceVisualizer:
    def __init__(self, robot_arm, interval=2.0, workspace_size=1.0):
        """
        Initialize the workspace visualizer
        
        Args:
            robot_arm (RobotArmIK): Robot arm instance
            interval (float): Time interval between cursor position updates (seconds)
            workspace_size (float): Size of the cubic workspace (meters)
        """
        self.robot_arm = robot_arm
        self.interval = interval
        self.workspace_size = workspace_size
        
        # Get the base model and data
        self.model = robot_arm.model
        self.data = robot_arm.data
        
        # Calculate workspace limits
        self.ws_center = np.zeros(3)  # Assuming first joint at center
        half_size = workspace_size / 2.0
        self.ws_min = self.ws_center - half_size
        self.ws_max = self.ws_center + half_size
        
        # Create a MuJoCo viewer
        self.viewer = None
        
        # Create the 3D cursor model
        self.create_cursor_model()
        
    def create_cursor_model(self):
        """Create MuJoCo model with 3D cursor made of intersecting capsules"""
        # Define the XML for a 3D cursor model
        cursor_xml = f"""
        <mujoco>
            <worldbody>
                <!-- X axis (red) -->
                <body name="x_capsule" pos="0 0 0">
                    <geom type="capsule" size="0.003 0.01" rgba="1 0 0 0.8" pos="0 0 0" quat="0.7071 0 0 0.7071"/>
                </body>
                <!-- Y axis (green) -->
                <body name="y_capsule" pos="0 0 0">
                    <geom type="capsule" size="0.003 0.01" rgba="0 1 0 0.8" pos="0 0 0" quat="0.7071 0 0.7071 0"/>
                </body>
                <!-- Z axis (blue) -->
                <body name="z_capsule" pos="0 0 0">
                    <geom type="capsule" size="0.003 0.01" rgba="0 0 1 0.8" pos="0 0 0.01" quat="1 0 0 0"/>
                </body>
            </worldbody>
        </mujoco>
        """
        # Create temporary file for the XML
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(cursor_xml.encode('utf-8'))
        
        # Load cursor model
        self.cursor_model = mujoco.MjModel.from_xml_path(temp_filename)
        self.cursor_data = mujoco.MjData(self.cursor_model)
        
        # Remove temporary file
        os.unlink(temp_filename)
        
        # Get body IDs for cursor components
        self.x_capsule_id = mujoco.mj_name2id(self.cursor_model, mujoco.mjtObj.mjOBJ_BODY, "x_capsule")
        self.y_capsule_id = mujoco.mj_name2id(self.cursor_model, mujoco.mjtObj.mjOBJ_BODY, "y_capsule")
        self.z_capsule_id = mujoco.mj_name2id(self.cursor_model, mujoco.mjtObj.mjOBJ_BODY, "z_capsule")
        
    def get_random_position(self):
        """Generate a random position within the workspace"""
        return np.random.uniform(self.ws_min, self.ws_max)
    
    def get_random_rotation(self):
        """Generate a random rotation matrix"""
        # Random unit quaternion
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        
        # Convert to rotation matrix
        return self.quat2mat(q)
    
    def quat2mat(self, quat):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def update_cursor_position(self, pos, rot_matrix):
        """Update the cursor's position and orientation"""
        # Convert rotation matrix to quaternion
        quat = self.rotation_matrix_to_quat(rot_matrix)
        
        # Set position for all capsules
        self.cursor_data.body(self.x_capsule_id).xpos = pos
        self.cursor_data.body(self.y_capsule_id).xpos = pos
        self.cursor_data.body(self.z_capsule_id).xpos = pos
        
        # Set orientation for all capsules
        self.cursor_data.body(self.x_capsule_id).xquat = quat
        self.cursor_data.body(self.y_capsule_id).xquat = quat
        self.cursor_data.body(self.z_capsule_id).xquat = quat
        
    def rotation_matrix_to_quat(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])
    
    def run_interactive_visualization(self):
        """Run the interactive visualization loop"""
        # Initialize MuJoCo viewer
        self.viewer = mjviewer.launch_passive(self.model, self.data)
        if not self.viewer:
            print("Failed to initialize MuJoCo viewer")
            return
            
        # Print instructions
        print("\nInteractive Visualization:")
        print("==========================")
        print(f"- Workspace: {self.workspace_size}m cube centered at origin")
        print(f"- Interval: {self.interval} seconds between position updates")
        print("- Red capsule: X axis")
        print("- Green capsule: Y axis")
        print("- Blue capsule: Z axis")
        print("- Press Esc or close the window to exit\n")
        
        # Main loop
        last_update_time = 0
        
        try:
            while self.viewer.is_running():
                current_time = time.time()
                
                # Update cursor position at the specified interval
                if current_time - last_update_time >= self.interval:
                    # Generate random position and orientation
                    random_pos = self.get_random_position()
                    random_rot = self.get_random_rotation()
                    
                    print(f"\nNew target: Position = {random_pos}, Rotation = {random_rot}")
                    
                    # Solve inverse kinematics
                    joint_angles, pos_error, orient_error, iterations = self.robot_arm.solve_ik(
                        random_pos, random_rot)
                    
                    print(f"IK solution: Iterations = {iterations}")
                    print(f"Errors: Position = {pos_error:.6f}m, Orientation = {orient_error:.6f}°")
                    
                    last_update_time = current_time
                    
                # Update robot visualization
                mujoco.mj_forward(self.model, self.data)
                
                # Update viewer
                self.viewer.sync()
                
                # Sleep to reduce CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
        finally:
            if self.viewer:
                self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description='Interactive IK Visualization for 6-DOF Robot Arm')
    parser.add_argument('mjcf_file', help='Path to the MuJoCo MJCF file')
    parser.add_argument('--interval', '-i', type=float, default=3.0, 
                        help='Time interval between cursor position updates (seconds)')
    parser.add_argument('--workspace', '-w', type=float, default=1.0,
                        help='Size of the cubic workspace in meters')
    parser.add_argument('--max-iterations', '-m', type=int, default=500,
                        help='Maximum iterations for IK solver')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-4,
                        help='Convergence tolerance for IK solver')
    parser.add_argument('--damping', '-d', type=float, default=0.1,
                        help='Damping factor for IK solver')
    
    args = parser.parse_args()
    
    # Initialize robot arm
    robot_arm = RobotArmIK(args.mjcf_file)
    
    # Initialize workspace visualizer
    visualizer = WorkspaceVisualizer(
        robot_arm, 
        interval=args.interval,
        workspace_size=args.workspace
    )
    
    # Run interactive visualization
    visualizer.run_interactive_visualization()

if __name__ == "__main__":
    main()