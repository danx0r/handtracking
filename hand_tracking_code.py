import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os
import pygame

def distance(a, b):
    d = 0
    for i in range(3):
        d += (a[i] - b[i]) ** 2
    return d ** 0.5

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

def process_camera_feed(period=4.0, camera_id=0, save_visualization=False, output_dir="."):
    """
    Periodically capture frames from a camera and detect hand landmarks.
    Display results in a Pygame window.
    
    Args:
        period (float): Time interval between frame captures in seconds
        camera_id (int): Camera device ID
        save_visualization (bool): Whether to save visualizations of the landmarks
        output_dir (str): Directory to save visualizations
        
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
                if landmarks_3d:
                    for i, hand_landmarks in enumerate(landmarks_3d):
                        print(f"Frame {frame_count} - Hand {i+1} landmarks:")
                        # for j, (x, y, z) in enumerate(hand_landmarks):
                        #     print(f"  Landmark {j}: ({x:.2f}, {y:.2f}, {z:.2f})")
                        print()  # Empty line between hands
                        d = distance(hand_landmarks[4], hand_landmarks[8])
                        # print ("DIST:", d)
                        grip = max(0, int(d/5)-10)
                        arm = i+1
                        # print ("ARM", arm, "GRIP", grip)
                        cmd = f'curl "http://localhost:8745/v1/arm{arm}/joints/set?j7={grip}"'
                        print (cmd)
                        os.system(cmd)
                else:
                    print(f"Frame {frame_count}: No hands detected")
                
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
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # Process a single image file
            landmarks_3d = detect_hand_landmarks_from_file(args.image, args.visualize, args.output)
            
            if not landmarks_3d:
                print("No hands detected in the image.")
                return
            
            # Print the 3D landmarks for each detected hand
            for i, hand_landmarks in enumerate(landmarks_3d):
                print(f"Hand {i+1} landmarks:")
                for j, (x, y, z) in enumerate(hand_landmarks):
                    print(f"  Landmark {j}: ({x:.2f}, {y:.2f}, {z:.2f})")
                print()  # Empty line between hands
                print ("thumb-first:", distance(hand_landmarks[4], hand_landmarks[8]))
                print ("first-second:", distance(hand_landmarks[12], hand_landmarks[8]))
        
        elif args.camera:
            # Process camera feed
            process_camera_feed(args.period, args.camera_id, args.visualize, args.output)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
