#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco
import math
from math import radians, degrees

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
        
    def solve_ik(self, target_pos, target_orient, max_iter=500, tol=1e-4, damping=0.1):
        """
        Solve inverse kinematics using damped least squares method
        
        Args:
            target_pos (list): Target position [x, y, z] in meters
            target_orient (list): Target orientation [pitch, roll, yaw] in degrees
            max_iter (int): Maximum iterations
            tol (float): Tolerance for convergence
            damping (float): Damping factor for numerical stability
            
        Returns:
            tuple: Joint angles in radians, position error in meters, orientation error in degrees
        """
        target_pos = np.array(target_pos)
        
        # Convert Euler angles to rotation matrix
        pitch, roll, yaw = [radians(angle) for angle in target_orient]
        
        # Create rotation matrix from Euler angles (ZYX convention)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        target_rot = Rz @ Ry @ Rx
        
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
            rel_rot = target_rot @ current_rot.T
            
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
            self.data.qpos += delta_q
            
            # Check joint limits
            for i in range(self.model.nq):
                if self.model.jnt_limited[i]:
                    self.data.qpos[i] = np.clip(self.data.qpos[i], 
                                               self.model.jnt_range[i, 0], 
                                               self.model.jnt_range[i, 1])
            
            iter_count += 1
        
        # Calculate final errors
        mujoco.mj_forward(self.model, self.data)
        final_pos = self.data.body(self.end_effector_id).xpos
        final_rot = self.data.body(self.end_effector_id).xmat.reshape(3, 3)
        
        pos_error = np.linalg.norm(target_pos - final_pos)
        
        rel_rot = target_rot @ final_rot.T
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
        angle = np.arccos((trace - 1) / 2)
        
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
                np.sqrt((R[0, 0] + 1) / 2),
                np.sqrt((R[1, 1] + 1) / 2) * np.sign(R[0, 1]),
                np.sqrt((R[2, 2] + 1) / 2) * np.sign(R[0, 2])
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

    def visualize(self, filename=None):
        """
        Visualize the current robot configuration
        
        Args:
            filename (str, optional): If provided, save the visualization to this file
        """
        # Create renderer
        renderer = mujoco.Renderer(self.model)
        
        # Update scene
        renderer.update_scene(self.data)
        
        # Render
        img = renderer.render()
        
        if filename:
            import imageio
            imageio.imwrite(filename, img)
        else:
            try:
                import matplotlib.pyplot as plt
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            except ImportError:
                print("Matplotlib not available for visualization")


def main():
    parser = argparse.ArgumentParser(description='Inverse Kinematics for 6-DOF Robot Arm')
    parser.add_argument('mjcf_file', help='Path to the MuJoCo MJCF file')
    parser.add_argument('x', type=float, help='X position in meters')
    parser.add_argument('y', type=float, help='Y position in meters')
    parser.add_argument('z', type=float, help='Z position in meters')
    parser.add_argument('pitch', type=float, help='Pitch angle in degrees')
    parser.add_argument('roll', type=float, help='Roll angle in degrees')
    parser.add_argument('yaw', type=float, help='Yaw angle in degrees')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the result')
    parser.add_argument('--output', '-o', type=str, help='Output image file (if visualizing)')
    parser.add_argument('--iterations', '-i', type=int, default=500, help='Maximum iterations')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-4, help='Convergence tolerance')
    parser.add_argument('--damping', '-d', type=float, default=0.1, help='Damping factor')
    
    args = parser.parse_args()
    
    # Initialize robot arm
    robot_arm = RobotArmIK(args.mjcf_file)
    
    # Solve IK
    target_pos = [args.x, args.y, args.z]
    target_orient = [args.pitch, args.roll, args.yaw]
    
    joint_angles, pos_error, orient_error, iterations = robot_arm.solve_ik(
        target_pos, 
        target_orient, 
        max_iter=args.iterations,
        tol=args.tolerance,
        damping=args.damping
    )
    
    # Print results
    print("Joint angles (radians):", joint_angles)
    print("Joint angles (degrees):", [degrees(angle) for angle in joint_angles])
    print(f"Position error: {pos_error:.6f} meters")
    print(f"Orientation error: {orient_error:.6f} degrees")
    print(f"Iterations: {iterations}")
    
    # Visualize if requested
    if args.visualize:
        robot_arm.visualize(args.output)

if __name__ == "__main__":
    main()