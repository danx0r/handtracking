#!/usr/bin/env python3
import argparse
import math
import numpy as np
from ikpy.chain import Chain
from ikpy.utils import geometry
import ikpy.utils.plot as plot_utils
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_mjcf_to_ikpy_chain(mjcf_file):
    """
    Parse an MJCF file and create an IKPy kinematic chain.
    
    This is a simplified implementation - a full parser would need to handle
    all MJCF elements and convert them to IKPy link objects.
    """
    try:
        # Since IKPy doesn't directly support MJCF, we'll use its URDF parsing capabilities
        # and assume the MJCF file has been converted to URDF format first
        # (you may need to implement a conversion from MJCF to URDF)
        
        # For demonstration purposes, we'll create a Chain object directly
        # In a real implementation, you would parse the MJCF file to extract joint information
        chain = Chain.from_urdf_file(mjcf_file)
        return chain
    except Exception as e:
        print(f"Error parsing MJCF file: {e}")
        print("Note: This example assumes the MJCF file is in URDF format or has been converted.")
        
        # For demonstration, create a 6-DOF arm chain if parsing fails
        print("Creating a default 6-DOF arm chain for demonstration...")
        return create_default_chain()

def create_default_chain():
    """Create a default 6-DOF arm chain for demonstration purposes."""
    from ikpy.link import URDFLink, OriginLink
    
    chain = Chain(name="default_arm", links=[
        OriginLink(),
        URDFLink(
            name="shoulder_pan_joint",
            origin_translation=[0, 0, 0.1],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-3.14, 3.14)
        ),
        URDFLink(
            name="shoulder_lift_joint",
            origin_translation=[0, 0, 0.2],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-3.14, 3.14)
        ),
        URDFLink(
            name="elbow_joint",
            origin_translation=[0, 0, 0.2],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-3.14, 3.14)
        ),
        URDFLink(
            name="wrist_1_joint",
            origin_translation=[0, 0, 0.2],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-3.14, 3.14)
        ),
        URDFLink(
            name="wrist_2_joint",
            origin_translation=[0, 0, 0.1],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-3.14, 3.14)
        ),
        URDFLink(
            name="wrist_3_joint",
            origin_translation=[0, 0, 0.1],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],
            bounds=(-3.14, 3.14)
        ),
        URDFLink(
            name="end_effector",
            origin_translation=[0, 0, 0.05],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
            bounds=(0, 0)
        )
    ])
    return chain

def degrees_to_radians(degrees):
    """Convert degrees to radians."""
    return degrees * (math.pi / 180)

def calculate_ik(chain, target_position, target_orientation):
    """
    Calculate inverse kinematics for the given target position and orientation.
    
    Args:
        chain: IKPy kinematic chain
        target_position: [x, y, z] in meters
        target_orientation: [pitch, roll, yaw] in radians
    
    Returns:
        joint_angles: List of joint angles in radians
    """
    # Convert Euler angles (pitch, roll, yaw) to rotation matrix
    rotation = Rotation.from_euler('xyz', target_orientation)
    rotation_matrix = rotation.as_matrix()
    
    # Create target orientation matrix (3x3)
    orientation_matrix = np.eye(4)
    orientation_matrix[:3, :3] = rotation_matrix
    
    # Add target position
    target_matrix = np.copy(orientation_matrix)
    target_matrix[:3, 3] = target_position
    
    # Calculate inverse kinematics
    joint_angles = chain.inverse_kinematics_frame(target_matrix, initial_position=None)
    
    return joint_angles

def calculate_fk(chain, joint_angles):
    """
    Calculate forward kinematics to get the end effector position and orientation.
    
    Args:
        chain: IKPy kinematic chain
        joint_angles: List of joint angles in radians
    
    Returns:
        position: [x, y, z] in meters
        orientation: [pitch, roll, yaw] in radians
    """
    # Get end effector transform matrix
    transform_matrix = chain.forward_kinematics(joint_angles)
    
    # Extract position
    position = transform_matrix[:3, 3]
    
    # Extract orientation (convert rotation matrix to Euler angles)
    rotation_matrix = transform_matrix[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    orientation = rotation.as_euler('xyz')
    
    return position, orientation

def calculate_error(target_position, target_orientation, actual_position, actual_orientation):
    """
    Calculate the error between target and actual position/orientation.
    
    Args:
        target_position: [x, y, z] in meters
        target_orientation: [pitch, roll, yaw] in radians
        actual_position: [x, y, z] in meters
        actual_orientation: [pitch, roll, yaw] in radians
    
    Returns:
        position_error: Error in meters
        orientation_error: Error in radians
    """
    # Calculate position error (Euclidean distance)
    position_error = np.linalg.norm(np.array(target_position) - np.array(actual_position))
    
    # Calculate orientation error (average angular difference)
    orientation_error = np.mean(np.abs(np.array(target_orientation) - np.array(actual_orientation)))
    
    return position_error, orientation_error

def visualize_robot(chain, joint_angles, target_position):
    """Visualize the robot and the target position."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the arm
    plot_utils.plot_robot(chain, joint_angles, ax, target=target_position)
    
    # Plot the target position
    ax.scatter(target_position[0], target_position[1], target_position[2], c='red', marker='o', s=100)
    
    # Set plot limits and labels
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title('Robot Arm IK Solution')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inverse Kinematics for 6-DOF Robot Arm')
    parser.add_argument('--mjcf', type=str, default=None, help='Path to MJCF file')
    parser.add_argument('--x', type=float, default=0.3, help='Target X position in meters')
    parser.add_argument('--y', type=float, default=0.0, help='Target Y position in meters')
    parser.add_argument('--z', type=float, default=0.5, help='Target Z position in meters')
    parser.add_argument('--pitch', type=float, default=0.0, help='Target pitch in degrees')
    parser.add_argument('--roll', type=float, default=0.0, help='Target roll in degrees')
    parser.add_argument('--yaw', type=float, default=0.0, help='Target yaw in degrees')
    parser.add_argument('--visualize', action='store_true', help='Visualize the robot')
    
    args = parser.parse_args()
    
    # Convert position to meters (already in meters from args)
    target_position = [args.x, args.y, args.z]
    
    # Convert orientation from degrees to radians
    target_orientation = [
        degrees_to_radians(args.pitch),
        degrees_to_radians(args.roll),
        degrees_to_radians(args.yaw)
    ]
    
    # Parse MJCF file or create default chain
    if args.mjcf:
        chain = parse_mjcf_to_ikpy_chain(args.mjcf)
    else:
        chain = create_default_chain()
    
    # Calculate inverse kinematics
    joint_angles = calculate_ik(chain, target_position, target_orientation)
    
    # Calculate forward kinematics to verify the solution
    actual_position, actual_orientation = calculate_fk(chain, joint_angles)
    
    # Calculate error
    position_error, orientation_error = calculate_error(
        target_position, target_orientation, actual_position, actual_orientation
    )
    
    # Print results
    print("\nTarget:")
    print(f"Position (meters): X={args.x}, Y={args.y}, Z={args.z}")
    print(f"Orientation (degrees): Pitch={args.pitch}, Roll={args.roll}, Yaw={args.yaw}")
    
    print("\nJoint Angles (radians):")
    for i, angle in enumerate(joint_angles):
        if i > 0:  # Skip the first joint (origin link)
            print(f"Joint {i}: {angle:.4f} rad ({math.degrees(angle):.2f} deg)")
    
    print("\nActual End Effector:")
    print(f"Position (meters): X={actual_position[0]:.4f}, Y={actual_position[1]:.4f}, Z={actual_position[2]:.4f}")
    print(f"Orientation (degrees): Pitch={math.degrees(actual_orientation[0]):.2f}, Roll={math.degrees(actual_orientation[1]):.2f}, Yaw={math.degrees(actual_orientation[2]):.2f}")
    
    print("\nError:")
    print(f"Position Error: {position_error:.4f} meters")
    print(f"Orientation Error: {orientation_error:.4f} rad ({math.degrees(orientation_error):.2f} degrees)")
    
    # Visualize if requested
    if args.visualize:
        visualize_robot(chain, joint_angles, target_position)

if __name__ == "__main__":
    main()