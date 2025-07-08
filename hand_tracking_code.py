import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os
import pygame
import math
from pygame import gfxdraw

def distance(a, b):
    d = 0
    for i in range(3):
        d += (a[i] - b[i]) ** 2
    return d ** 0.5

def calculate_hand_segment_length_sum(hand_landmarks):
    """Calculate the sum of 3D distances between all connected landmarks (20 segments total)."""
    # Hand connections (MediaPipe hand connections)
    hand_connections = [
        # Wrist to fingers
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index finger
        (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20)
    ]
    
    total_length = 0
    for start_idx, end_idx in hand_connections:
        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
            segment_length = distance(hand_landmarks[start_idx], hand_landmarks[end_idx])
            total_length += segment_length
    
    return total_length

def calculate_proxy_z_from_segment_length(total_segment_length):
    """
    Calculate a proxy Z coordinate (distance from camera) based on total segment length.
    Uses reciprocal relationship: larger segment length = closer to camera = higher Z value.
    
    Args:
        total_segment_length: Sum of all hand segment lengths
        
    Returns:
        Proxy Z coordinate using reciprocal formula
    """
    # Avoid division by zero
    if total_segment_length <= 0:
        return 500  # Return a default far distance
    
    # Calibration constant - adjust this based on your setup
    # Higher K = higher Z values overall
    # Based on observed range: 1400 (far) to 4000 (close) for segment lengths
    K = 4200000  # Adjusted for better range mapping
    
    # Calculate reciprocal relationship: K * (1.0 / distance)
    # Larger segment length (closer to camera) = higher Z value
    proxy_z = K * (1.0 / total_segment_length)
    
    return proxy_z

def calculate_hand_position(hand_landmarks, proxy_z=None):
    """
    Calculate the position of the hand using the wrist as the reference point.
    
    Args:
        hand_landmarks: List of 21 hand landmarks [(x, y, z), ...]
        proxy_z: Optional proxy Z distance to use instead of landmark z
        
    Returns:
        tuple: (x, y, z) position of the hand (wrist position)
    """
    if not hand_landmarks or len(hand_landmarks) < 21:
        return None
    
    # Use wrist landmark (index 0) as the hand position
    wrist_x, wrist_y, wrist_z = hand_landmarks[0]
    
    # Use proxy Z if provided, otherwise use landmark z
    if proxy_z is not None:
        wrist_z = proxy_z
    
    return (wrist_x, wrist_y, wrist_z)

def calculate_hand_center(hand_landmarks, proxy_z=None):
    """
    Calculate the center position of the hand based on palm landmarks.
    
    Args:
        hand_landmarks: List of 21 hand landmarks [(x, y, z), ...]
        proxy_z: Optional proxy Z distance to use instead of landmark z
        
    Returns:
        tuple: (x, y, z) center position of the hand
    """
    if not hand_landmarks or len(hand_landmarks) < 21:
        return None
    
    # Use palm landmarks (wrist + finger bases) to calculate center
    palm_indices = [0, 1, 5, 9, 13, 17]  # wrist + finger bases
    
    center_x = sum(hand_landmarks[i][0] for i in palm_indices) / len(palm_indices)
    center_y = sum(hand_landmarks[i][1] for i in palm_indices) / len(palm_indices)
    center_z = sum(hand_landmarks[i][2] for i in palm_indices) / len(palm_indices)
    
    # Use proxy Z if provided, otherwise use calculated z
    if proxy_z is not None:
        center_z = proxy_z
    
    return (center_x, center_y, center_z)

def calculate_hand_orientation(hand_landmarks):
    """
    Calculate the orientation of the hand in terms of yaw, pitch, and roll angles.
    
    Args:
        hand_landmarks: List of 21 hand landmarks [(x, y, z), ...]
        
    Returns:
        tuple: (yaw, pitch, roll) angles in radians
    """
    if not hand_landmarks or len(hand_landmarks) < 21:
        return None
    
    # Key landmarks for orientation calculation
    wrist = np.array(hand_landmarks[0])          # Wrist
    middle_mcp = np.array(hand_landmarks[9])     # Middle finger MCP joint
    index_mcp = np.array(hand_landmarks[5])      # Index finger MCP joint
    pinky_mcp = np.array(hand_landmarks[17])     # Pinky MCP joint
    
    # Calculate vectors
    palm_vector = middle_mcp - wrist  # Vector from wrist to middle finger base
    side_vector = index_mcp - pinky_mcp  # Vector across the palm
    
    # Normalize vectors
    palm_vector = palm_vector / np.linalg.norm(palm_vector)
    side_vector = side_vector / np.linalg.norm(side_vector)
    
    # Calculate normal vector (perpendicular to palm)
    normal_vector = np.cross(palm_vector, side_vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Calculate yaw (rotation around z-axis)
    yaw = math.atan2(palm_vector[1], palm_vector[0])
    
    # Calculate pitch (rotation around y-axis)
    pitch = math.asin(-palm_vector[2])
    
    # Calculate roll (rotation around x-axis)
    roll = math.atan2(normal_vector[1], normal_vector[2])
    
    return (yaw, pitch, roll)

def print_hand_data(landmarks_3d, frame_count=None, include_landmarks=False, send_robot_commands=False, landmark_filter='all'):
    """
    Print hand data to stdout in a consistent format across all modes.
    
    Args:
        landmarks_3d: List of hand landmarks for each detected hand
        frame_count: Optional frame number for camera mode
        include_landmarks: Whether to print individual landmark coordinates
        send_robot_commands: Whether to send commands to robot arms
        landmark_filter: Either 'all' or comma-separated list of landmark indices to print
    """
    if not landmarks_3d:
        if frame_count is not None:
            print(f"Frame {frame_count}: no hands detected")
        else:
            print("No hands detected")
        return
    
    # Sort hands for consistent ordering when there are 2 hands
    if len(landmarks_3d) == 2:
        if landmarks_3d[0][0][0] > landmarks_3d[1][0][0]:  # swap so left hand is always first
            landmarks_3d = landmarks_3d[::-1]
    
    for i, hand_landmarks in enumerate(landmarks_3d):
        # Print individual landmark coordinates if requested
        if include_landmarks:
            # Parse landmark filter
            if landmark_filter.lower() == 'all':
                landmarks_to_print = range(len(hand_landmarks))
            elif landmark_filter.lower() == 'none':
                landmarks_to_print = []
            else:
                try:
                    landmarks_to_print = []
                    for item in landmark_filter.split(','):
                        item = item.strip()
                        if '-' in item:
                            # Handle range like "0-4"
                            start, end = item.split('-', 1)
                            start, end = int(start.strip()), int(end.strip())
                            landmarks_to_print.extend(range(start, end + 1))
                        else:
                            # Handle single number
                            landmarks_to_print.append(int(item))
                except ValueError:
                    print(f"Warning: Invalid landmark filter '{landmark_filter}', showing no landmarks")
                    landmarks_to_print = []
            
            # Only print header if there are landmarks to show
            if landmarks_to_print:
                if frame_count is not None:
                    print(f"Frame {frame_count} - Hand {i+1} landmarks:")
                else:
                    print(f"Hand {i+1} landmarks:")
            
            for j, (x, y, z) in enumerate(hand_landmarks):
                if j in landmarks_to_print:
                    print(f"  Landmark {j}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        print()  # Empty line between hands
        
        # Calculate and print distances - consistent format for all modes
        thumb_first_dist = distance(hand_landmarks[4], hand_landmarks[8])
        first_second_dist = distance(hand_landmarks[12], hand_landmarks[8])
        total_segment_length = calculate_hand_segment_length_sum(hand_landmarks)
        
        # print("DIST:", thumb_first_dist)
        # print("thumb-first:", thumb_first_dist)
        # print("first-second:", first_second_dist)
        print("TOTAL_SEGMENT_LENGTH:", total_segment_length)
        
        # Calculate and print proxy Z coordinate (distance from camera)
        proxy_z = calculate_proxy_z_from_segment_length(total_segment_length)
        print("PROXY_Z_DISTANCE:", proxy_z)
        
        # Calculate and print hand position
        hand_position = calculate_hand_position(hand_landmarks, proxy_z)
        if hand_position:
            print(f"HAND_POSITION (wrist): ({hand_position[0]:.2f}, {hand_position[1]:.2f}, {hand_position[2]:.2f})")
        
        # Calculate and print hand center
        hand_center = calculate_hand_center(hand_landmarks, proxy_z)
        if hand_center:
            print(f"HAND_CENTER (palm): ({hand_center[0]:.2f}, {hand_center[1]:.2f}, {hand_center[2]:.2f})")
        
        # Calculate and print hand orientation
        orientation = calculate_hand_orientation(hand_landmarks)
        if orientation:
            yaw, pitch, roll = orientation
            # Convert radians to degrees for easier reading
            yaw_deg = math.degrees(yaw)
            pitch_deg = math.degrees(pitch)
            roll_deg = math.degrees(roll)
            print(f"HAND_ORIENTATION (yaw, pitch, roll): ({yaw_deg:.1f}°, {pitch_deg:.1f}°, {roll_deg:.1f}°)")
            print(f"HAND_ORIENTATION (radians): ({yaw:.3f}, {pitch:.3f}, {roll:.3f})")
        
        # Robot control logic
        if send_robot_commands:
            grip = max(0, 12-int(thumb_first_dist/8))
            arm = i+1
            print("ARM", arm, "GRIP", grip)
            cmd = f'curl "http://localhost:8745/v1/arm{arm}/joints/set?j7={grip}"'
            print(cmd)
            os.system(cmd)
        
        # Add separator between hands (except for the last hand)
        if i < len(landmarks_3d) - 1:
            print("~" * 40)

def apply_local_rotation(point_3d, center_3d, angle_y, angle_x):
    """Apply rotation around local center of the landmark set."""
    x, y, z = point_3d
    cx, cy, cz = center_3d
    
    # Translate to origin (relative to local center)
    x_local = x - cx
    y_local = y - cy
    z_local = z - cz
    
    # Apply Y axis rotation (yaw - left/right mouse)
    cos_y = math.cos(angle_y)
    sin_y = math.sin(angle_y)
    x_rot_y = x_local * cos_y + z_local * sin_y
    y_rot_y = y_local
    z_rot_y = -x_local * sin_y + z_local * cos_y
    
    # Apply X axis rotation (pitch - up/down mouse)
    cos_x = math.cos(angle_x)
    sin_x = math.sin(angle_x)
    x_rot = x_rot_y
    y_rot = y_rot_y * cos_x - z_rot_y * sin_x
    z_rot = y_rot_y * sin_x + z_rot_y * cos_x
    
    # Translate back to world space
    x_final = x_rot + cx
    y_final = y_rot + cy
    z_final = z_rot + cz
    
    return (x_final, y_final, z_final)

def project_3d_to_2d(point_3d, camera_pos, screen_width, screen_height):
    """Project 3D point to 2D screen coordinates with perspective."""
    x, y, z = point_3d
    
    # Translate by camera position
    x_cam = x - camera_pos[0]
    y_cam = y - camera_pos[1]
    z_cam = z - camera_pos[2]
    
    # Perspective projection
    focal_length = 1000
    if abs(z_cam) < 1:
        z_cam = 1 if z_cam >= 0 else -1  # Avoid division by zero
    
    screen_x = int(screen_width / 2 + (focal_length * x_cam / z_cam))
    screen_y = int(screen_height / 2 - (focal_length * y_cam / z_cam))
    
    return screen_x, screen_y, z_cam

def draw_sphere(screen, center, radius, color, z_depth):
    """Draw a sphere using concentric circles with depth-based shading."""
    if z_depth <= 0:
        return
    
    # Adjust radius and color based on depth
    depth_factor = max(0.3, 1.0 / (1.0 + z_depth * 0.001))
    adj_radius = max(2, int(radius * depth_factor))
    adj_color = tuple(int(c * depth_factor) for c in color)
    
    # Draw filled circle
    pygame.draw.circle(screen, adj_color, center, adj_radius)
    
    # Add highlight for 3D effect
    highlight_color = tuple(min(255, int(c * 1.5)) for c in adj_color)
    highlight_radius = max(1, adj_radius // 3)
    highlight_center = (center[0] - adj_radius // 3, center[1] - adj_radius // 3)
    pygame.draw.circle(screen, highlight_color, highlight_center, highlight_radius)

def draw_cylinder(screen, start_2d, end_2d, radius, color, z_depth_avg):
    """Draw a cylinder as a thick line with depth-based shading."""
    if z_depth_avg <= 0:
        return
    
    # Adjust thickness and color based on depth
    depth_factor = max(0.3, 1.0 / (1.0 + z_depth_avg * 0.001))
    adj_radius = max(1, int(radius * depth_factor))
    adj_color = tuple(int(c * depth_factor) for c in color)
    
    # Draw thick line
    pygame.draw.line(screen, adj_color, start_2d, end_2d, adj_radius * 2)

def visualize_3d_landmarks(landmarks_3d_list, screen_width=1200, screen_height=800, background_image=None):
    """3D visualization of hand landmarks using pygame."""
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("3D Hand Landmarks Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    
    # Prepare landmarks for 3D visualization with proper 2D alignment
    # We need to transform the landmarks so they appear in the right 2D positions initially
    aligned_landmarks = []
    
    # Get background image dimensions if available
    bg_width, bg_height = screen_width, screen_height
    if background_image is not None:
        bg_height, bg_width = background_image.shape[:2]
    
    for hand_landmarks in landmarks_3d_list:
        if not hand_landmarks:
            continue
            
        # Transform landmarks for initial 2D alignment
        aligned_hand = []
        for x, y, z in hand_landmarks:
            # Convert pixel coordinates to centered coordinate system
            # Map from image coordinates to screen coordinates 
            centered_x = x - (bg_width / 2)
            centered_y = (bg_height / 2) - y  # Flip Y axis
            
            # Scale to fit screen nicely
            scale_factor = min(screen_width / bg_width, screen_height / bg_height)
            
            # Apply the transformation
            aligned_x = centered_x * scale_factor
            aligned_y = centered_y * scale_factor
            aligned_z = z * scale_factor  # Keep relative depth
            
            aligned_hand.append((aligned_x, aligned_y, aligned_z))
        
        aligned_landmarks.append(aligned_hand)
    
    
    # Calculate bounding box center of all landmarks for local rotation
    landmark_center = [0, 0, 0]
    total_landmarks = 0
    
    for hand_landmarks in aligned_landmarks:
        for x, y, z in hand_landmarks:
            landmark_center[0] += x
            landmark_center[1] += y
            landmark_center[2] += z
            total_landmarks += 1
    
    if total_landmarks > 0:
        landmark_center[0] /= total_landmarks
        landmark_center[1] /= total_landmarks
        landmark_center[2] /= total_landmarks
    
    
    # Prepare background image if provided
    background_surface = None
    if background_image is not None:
        # Convert OpenCV image (BGR) to pygame surface
        background_rgb = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        # Resize to fit screen while maintaining aspect ratio
        img_height, img_width = background_rgb.shape[:2]
        scale_x = screen_width / img_width
        scale_y = screen_height / img_height
        scale = min(scale_x, scale_y)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        background_resized = cv2.resize(background_rgb, (new_width, new_height))
        
        # Convert to pygame surface
        background_surface = pygame.surfarray.make_surface(background_resized.swapaxes(0, 1))
        # Make it translucent
        background_surface.set_alpha(128)  # 50% transparent
    
    # Camera controls - start with proper zoom for 2D alignment
    camera_pos = [0, 0, -1150]  # Start even further back for proper scale
    camera_angle_y = 0  # Rotation around Y axis (vertical, up-down)
    camera_angle_x = 0  # Rotation around X axis (horizontal, left-right)
    left_mouse_dragging = False
    right_mouse_dragging = False
    last_mouse_pos = (0, 0)
    
    # Color scheme matching MediaPipe's default styles
    landmark_colors = {
        'wrist': (255, 255, 255),
        'thumb': (255, 128, 0),
        'index': (255, 255, 0),
        'middle': (0, 255, 0),
        'ring': (0, 255, 255),
        'pinky': (255, 0, 255)
    }
    
    connection_color = (128, 128, 128)
    
    # Hand connections (MediaPipe hand connections)
    hand_connections = [
        # Wrist to fingers
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index finger
        (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20)
    ]
    
    # Landmark to finger mapping
    landmark_to_finger = {
        0: 'wrist',
        1: 'thumb', 2: 'thumb', 3: 'thumb', 4: 'thumb',
        5: 'index', 6: 'index', 7: 'index', 8: 'index',
        9: 'middle', 10: 'middle', 11: 'middle', 12: 'middle',
        13: 'ring', 14: 'ring', 15: 'ring', 16: 'ring',
        17: 'pinky', 18: 'pinky', 19: 'pinky', 20: 'pinky'
    }
    
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_LEFT:
                    camera_angle_y -= 0.1
                elif event.key == pygame.K_RIGHT:
                    camera_angle_y += 0.1
                elif event.key == pygame.K_UP:
                    camera_pos[2] += 10
                elif event.key == pygame.K_DOWN:
                    camera_pos[2] -= 10
                elif event.key == pygame.K_w:
                    camera_pos[1] += 10
                elif event.key == pygame.K_s:
                    camera_pos[1] -= 10
                elif event.key == pygame.K_a:
                    camera_pos[0] -= 10
                elif event.key == pygame.K_d:
                    camera_pos[0] += 10
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button - rotation
                    left_mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 3:  # Right mouse button - panning
                    right_mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    left_mouse_dragging = False
                elif event.button == 3:  # Right mouse button
                    right_mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if left_mouse_dragging or right_mouse_dragging:
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - last_mouse_pos[0]
                    dy = mouse_pos[1] - last_mouse_pos[1]
                    
                    if left_mouse_dragging:
                        # Rotate around Y axis (yaw) and X axis (pitch)
                        camera_angle_y += dx * 0.01  # Horizontal mouse = Y rotation (yaw)
                        camera_angle_x += dy * 0.01  # Vertical mouse = X rotation (pitch)
                    elif right_mouse_dragging:
                        # Pan camera left/right and up/down
                        camera_pos[0] -= dx * 0.5  # Horizontal pan
                        camera_pos[1] += dy * 0.5  # Vertical pan
                    
                    last_mouse_pos = mouse_pos
            elif event.type == pygame.MOUSEWHEEL:
                # Mouse wheel for zoom
                camera_pos[2] += event.y * 20
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw background image if available
        if background_surface:
            # Center the background image on screen
            bg_rect = background_surface.get_rect()
            bg_rect.center = (screen_width // 2, screen_height // 2)
            screen.blit(background_surface, bg_rect)
        
        # Process each hand
        for hand_idx, hand_landmarks in enumerate(aligned_landmarks):
            if not hand_landmarks:
                continue
            
            # Apply local rotation around landmark center, then project to 2D
            projected_landmarks = []
            for i, landmark in enumerate(hand_landmarks):
                # Apply local rotation around the landmark center
                rotated_landmark = apply_local_rotation(landmark, landmark_center, camera_angle_y, camera_angle_x)
                
                # Project to 2D
                x2d, y2d, z_depth = project_3d_to_2d(rotated_landmark, camera_pos, screen_width, screen_height)
                projected_landmarks.append((x2d, y2d, z_depth))
                
            
            # Draw connections (cylinders)
            for start_idx, end_idx in hand_connections:
                if start_idx < len(projected_landmarks) and end_idx < len(projected_landmarks):
                    start_2d = projected_landmarks[start_idx][:2]
                    end_2d = projected_landmarks[end_idx][:2]
                    z_depth_avg = (projected_landmarks[start_idx][2] + projected_landmarks[end_idx][2]) / 2
                    
                    if z_depth_avg > 0:
                        draw_cylinder(screen, start_2d, end_2d, 6, connection_color, z_depth_avg)
            
            # Draw landmarks (spheres)
            for i, (x2d, y2d, z_depth) in enumerate(projected_landmarks):
                # More lenient bounds checking and always draw something visible
                if 0 <= x2d < screen_width and 0 <= y2d < screen_height:
                    finger_type = landmark_to_finger.get(i, 'wrist')
                    color = landmark_colors.get(finger_type, (255, 255, 255))
                    
                    # Different sizes for different landmark types (doubled)
                    if i == 0:  # Wrist
                        radius = 16
                    elif i in [4, 8, 12, 16, 20]:  # Fingertips
                        radius = 12
                    else:  # Joint landmarks
                        radius = 8
                    
                    # Simple fallback - draw as regular circle if depth rendering fails
                    if z_depth > 0:
                        draw_sphere(screen, (x2d, y2d), radius, color, z_depth)
                    else:
                        # Fallback: draw simple circle
                        pygame.draw.circle(screen, color, (x2d, y2d), radius)
        
        # Draw UI
        controls_text = [
            "3D Hand Landmarks Visualization",
            "Left mouse: Rotate, Right mouse: Pan",
            "Mouse wheel: Zoom",
            "Arrow keys: Rotate, WASD: Move",
            "Q/ESC: Quit"
        ]
        
        for i, text in enumerate(controls_text):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (10, 10 + i * 25))
        
        # Show number of hands
        hands_text = f"Hands detected: {len(aligned_landmarks)}"
        text_surface = font.render(hands_text, True, (0, 255, 0))
        screen.blit(text_surface, (10, screen_height - 30))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

def detect_hand_landmarks_from_image(rgb_image, save_visualization=False, output_path=None):
    """
    Detect 3D hand landmarks from a raw RGB image.
    
    Args:
        rgb_image (numpy.ndarray): Input RGB image as a numpy array with shape (height, width, 3)
        save_visualization (bool): Whether to save a visualization of the landmarks
        output_path (str, optional): Path to save the visualization (if save_visualization is True)
        
    Returns:
        tuple: (list of detected landmarks in 3D space for each hand [[(x, y, z), ...], ...],
               visualization image if save_visualization is True, otherwise None)
    """
    # Initialize MediaPipe Hand solution
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Get image dimensions
    height, width = rgb_image.shape[:2]
    
    # Process the RGB image with MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        results = hands.process(rgb_image)
        
        # Initialize an empty list to store landmarks
        all_landmarks_3d = []
        
        # Create a copy of the image for visualization
        vis_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Process each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 3D landmarks
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    # MediaPipe normalizes coordinates, so we'll multiply by image dimensions
                    # The z-coordinate represents depth (relative to wrist)
                    x = landmark.x * width
                    y = landmark.y * height
                    z = landmark.z * width  # Scale z similarly to x
                    landmarks_3d.append((x, y, z))
                
                all_landmarks_3d.append(landmarks_3d)
                
                # Draw the hand annotations on the image
                mp_drawing.draw_landmarks(
                    vis_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Add status text to the visualization
        if all_landmarks_3d:
            status_text = f"{len(all_landmarks_3d)} hand(s) detected"
        else:
            status_text = "No hands detected"
        
        cv2.putText(vis_image, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save the annotated image if visualization is requested
        if save_visualization and output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to {output_path}")
        
        return all_landmarks_3d, vis_image

def detect_hand_landmarks_from_file(image_path, save_visualization=False, output_path="hand_landmarks.jpg"):
    """
    Detect 3D hand landmarks from an image file.
    
    Args:
        image_path (str): Path to the input image file
        save_visualization (bool): Whether to save a visualization of the landmarks
        output_path (str): Path to save the visualization (if save_visualization is True)
        
    Returns:
        list: List of detected landmarks in 3D space for each hand [[(x, y, z), ...], ...]
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Call the function that processes raw RGB image
    landmarks_3d, _ = detect_hand_landmarks_from_image(rgb_image, save_visualization, output_path)
    return landmarks_3d

def process_camera_feed(period=4.0, camera_id=0, save_visualization=False, output_dir=".", send_robot_commands=False, landmark_filter='all'):
    """
    Periodically capture frames from a camera and detect hand landmarks.
    Display results in a Pygame window.
    
    Args:
        period (float): Time interval between frame captures in seconds
        camera_id (int): Camera device ID
        save_visualization (bool): Whether to save visualizations of the landmarks
        output_dir (str): Directory to save visualizations
        send_robot_commands (bool): Whether to send commands to robot arms
        landmark_filter (str): Either 'all' or comma-separated list of landmark indices to print
        
    Returns:
        None
    """
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera with ID {camera_id}")
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Hand Tracking")
    screen = pygame.display.set_mode((frame_width, frame_height))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Create output directory if it doesn't exist
    if save_visualization and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Camera initialized. Capturing frames every {period} seconds.")
    print("Close the window or press 'q' to quit.")
    
    frame_count = 0
    last_capture_time = 0
    current_frame = None
    last_processed_frame = None
    running = True
    
    try:
        while running:
            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            # Capture frame (continuously)
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Always store the most recent frame
            current_frame = frame.copy()
            
            # Check if it's time to process a new frame
            current_time = time.time()
            if current_time - last_capture_time >= period:
                print ("=" * 44)
                # Convert to RGB for processing
                rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                
                # Generate output path for this frame if visualization is enabled
                output_path = None
                if save_visualization:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = f"{output_dir}/hand_landmarks_{timestamp}_{frame_count}.jpg"
                
                # Process the frame
                landmarks_3d, last_processed_frame = detect_hand_landmarks_from_image(
                    rgb_frame, save_visualization, output_path)
                
                # Print landmark information
                print_hand_data(landmarks_3d, frame_count=frame_count, include_landmarks=True, send_robot_commands=send_robot_commands, landmark_filter=landmark_filter)
                
                # Update timing and frame count
                last_capture_time = current_time
                frame_count += 1
            
            # Display the frame
            if last_processed_frame is not None:
                # Convert OpenCV BGR image to Pygame RGB format
                processed_frame_rgb = cv2.cvtColor(last_processed_frame, cv2.COLOR_BGR2RGB)
                pygame_surface = pygame.surfarray.make_surface(processed_frame_rgb.swapaxes(0, 1))
                screen.blit(pygame_surface, (0, 0))
                
                # Add additional text with Pygame
                next_capture_text = f"Next capture in: {max(0, period - (time.time() - last_capture_time)):.1f}s"
                text_surface = font.render(next_capture_text, True, (0, 255, 0))
                screen.blit(text_surface, (10, 70))
                
                quit_text = "Press 'q' to quit"
                text_surface = font.render(quit_text, True, (0, 255, 0))
                screen.blit(text_surface, (10, frame_height - 40))
            
            # Update display
            pygame.display.flip()
            
            # Control the frame rate (for UI responsiveness)
            clock.tick(30)  # 30 FPS for the display
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        cap.release()
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Detect 3D hand landmarks from an image or camera feed')
    
    # Create a mutually exclusive group for input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str, help='Path to the input image file')
    input_group.add_argument('--camera', '-c', action='store_true', help='Use camera as input')
    
    # Camera options
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--period', '-p', type=float, default=4.0, 
                       help='Time interval between camera captures in seconds (default: 4.0)')
    
    # Visualization options
    parser.add_argument('--visualize', '-v', action='store_true', help='Save visualization of landmarks')
    parser.add_argument('--output', '-o', type=str, default='hand_landmarks.jpg', 
                       help='Path to save visualization for image mode or directory for camera mode')
    parser.add_argument('--3d', action='store_true', help='Launch 3D visualization window with spheres and cylinders')
    parser.add_argument('--robot', action='store_true', help='Enable robot arm control commands')
    parser.add_argument('--landmarks', type=str, default='none', 
                       help='Comma-separated list of landmark indices to print, supports ranges like 0-4 (default: none)')
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # Process a single image file
            landmarks_3d = detect_hand_landmarks_from_file(args.image, args.visualize, args.output)
            
            if not landmarks_3d:
                print("No hands detected in the image.")
                return
            
            # Launch 3D visualization if requested
            if getattr(args, '3d', False):
                # Print hand data to stdout including landmark coordinates
                print_hand_data(landmarks_3d, include_landmarks=True, landmark_filter=args.landmarks)
                
                # Load the original image for background
                background_image = cv2.imread(args.image)
                visualize_3d_landmarks(landmarks_3d, background_image=background_image)
            else:
                # Print the 3D landmarks for each detected hand
                print_hand_data(landmarks_3d, include_landmarks=True, landmark_filter=args.landmarks)
        
        elif args.camera:
            # Process camera feed
            if getattr(args, '3d', False):
                print("3D visualization is only supported for image files, not camera feed.")
                return
            process_camera_feed(args.period, args.camera_id, args.visualize, args.output, args.robot, args.landmarks)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
