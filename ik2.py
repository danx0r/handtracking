import mujoco
import numpy as np
from scipy.optimize import minimize

class RobotArmIK:
    def __init__(self, mjcf_path):
        """
        Initialize the robot arm with a MuJoCo MJCF file.
        
        Args:
            mjcf_path (str): Path to the MJCF file for the 6-DOF robot arm
        """
        # Load the MJCF model
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        # Identify the end effector and joint indices
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        self.joint_ids = []
        for i in range(6):  # Assuming 6 joints for 6-DOF arm
            print ("I", i)
            joint_name = f"joint{i+1}"  # Adjust naming convention as needed
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in model")
            self.joint_ids.append(joint_id)
        
        # Set joint limits
        self.joint_limits = []
        for joint_id in self.joint_ids:
            lower = self.model.jnt_range[joint_id][0]
            upper = self.model.jnt_range[joint_id][1]
            self.joint_limits.append((lower, upper))
    
    def forward_kinematics(self, joint_angles):
        """
        Compute the end effector pose given joint angles.
        
        Args:
            joint_angles (np.ndarray): Array of 6 joint angles in radians
            
        Returns:
            tuple: (x, y, z, roll, pitch, yaw) of end effector
        """
        # Set joint positions
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = joint_angles[i]
        
        # Forward the simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Get end effector position
        pos = self.data.body(self.end_effector_id).xpos.copy()
        
        # Get end effector orientation (convert from quaternion to euler angles)
        quat = self.data.body(self.end_effector_id).xquat.copy()
        euler = self._quat_to_euler(quat)
        
        return np.concatenate([pos, euler])
    
    def _quat_to_euler(self, quat):
        """Convert quaternion to euler angles (roll, pitch, yaw)"""
        # Extract quaternion components
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.pi / 2.0 * np.sign(sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _cost_function(self, joint_angles, target_pose):
        """
        Cost function for optimization: distance between current and target pose.
        
        Args:
            joint_angles (np.ndarray): Array of 6 joint angles
            target_pose (np.ndarray): Target pose [x, y, z, roll, pitch, yaw]
            
        Returns:
            float: Cost value (sum of squared differences)
        """
        current_pose = self.forward_kinematics(joint_angles)
        
        # Position error (squared distance)
        pos_error = np.sum((current_pose[:3] - target_pose[:3])**2)
        
        # Orientation error (squared angular difference)
        # Note: angular differences need special handling for wrapping
        orient_error = 0
        for i in range(3):
            diff = self._angular_diff(current_pose[3+i], target_pose[3+i])
            orient_error += diff**2
        
        # Weight position and orientation errors
        # Adjust these weights based on your specific requirements
        position_weight = 1.0
        orientation_weight = 0.5
        
        return position_weight * pos_error + orientation_weight * orient_error
    
    def _angular_diff(self, angle1, angle2):
        """Calculate the shortest angular difference between two angles"""
        diff = (angle1 - angle2) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        return diff
    
    def inverse_kinematics(self, x, y, z, roll, pitch, yaw, initial_guess=None):
        """
        Solve inverse kinematics to find joint angles for a target end effector pose.
        
        Args:
            x, y, z: Target position coordinates
            roll, pitch, yaw: Target orientation angles (in radians)
            initial_guess: Initial joint angles for optimization (optional)
            
        Returns:
            np.ndarray: Array of 6 joint angles that achieve the target pose
        """
        target_pose = np.array([x, y, z, roll, pitch, yaw])
        
        # Use current joint positions as initial guess if not provided
        if initial_guess is None:
            initial_guess = np.array([self.data.qpos[joint_id] for joint_id in self.joint_ids])
        
        # Define bounds for each joint based on the joint limits
        bounds = self.joint_limits
        
        # Run the optimization
        result = minimize(
            self._cost_function,
            initial_guess,
            args=(target_pose,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        # Check if optimization was successful
        if not result.success:
            print(f"Warning: IK optimization did not converge: {result.message}")
        
        # Final joint angles
        joint_angles = result.x
        
        # Compute the achieved pose
        achieved_pose = self.forward_kinematics(joint_angles)
        position_error = np.linalg.norm(achieved_pose[:3] - target_pose[:3])
        
        print(f"IK solved with position error: {position_error:.6f} units")
        
        return joint_angles

# Example usage
if __name__ == "__main__":
    # Path to your MJCF file
    mjcf_path = "arm2.xml"
    
    # Initialize the robot arm
    robot = RobotArmIK(mjcf_path)
    
    # Target pose: x, y, z, roll, pitch, yaw
    target_x, target_y, target_z = 0, 0, 0 
    # target_roll, target_pitch, target_yaw = 0.0, np.pi/4, np.pi/2
    target_roll, target_pitch, target_yaw = 0.0, 0, 0
    
    # Solve inverse kinematics
    joint_angles = robot.inverse_kinematics(
        target_x, target_y, target_z,
        target_roll, target_pitch, target_yaw
    )
    
    print("Joint angles solution:")
    for i, angle in enumerate(joint_angles):
        print(f"Joint {i+1}: {angle:.4f} rad ({angle * 180/np.pi:.2f} deg)")
    
    # Verify the solution with forward kinematics
    achieved_pose = robot.forward_kinematics(joint_angles)
    print("\nAchieved end effector pose:")
    print(f"Position: x={achieved_pose[0]:.4f}, y={achieved_pose[1]:.4f}, z={achieved_pose[2]:.4f}")
    print(f"Orientation: roll={achieved_pose[3]:.4f}, pitch={achieved_pose[4]:.4f}, yaw={achieved_pose[5]:.4f}")
