import cv2
import mediapipe as mp
import numpy as np
import argparse
import time

def detect_hand_landmarks_from_image(rgb_image, save_visualization=False, output_path=None):
    """
    Detect 3D hand landmarks from a raw RGB image.
    
    Args:
        rgb_image (numpy.ndarray): Input RGB image as a numpy array with shape (height, width, 3)
        save_visualization (bool): Whether to save a visualization of the landmarks
        output_path (str, optional): Path to save the visualization (if save_visualization is True)
        
    Returns:
        list: List of detected landmarks in 3D space for each hand [[(x, y, z), ...], ...]
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
        
        # Create a copy of the image for visualization if needed
        vis_image = None
        if save_visualization:
            # Convert RGB to BGR for OpenCV visualization and saving
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
                
                # Draw the hand annotations on the image if visualization is requested
                if save_visualization and vis_image is not None:
                    mp_drawing.draw_landmarks(
                        vis_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        
        # Save the annotated image if visualization is requested
        if save_visualization and results.multi_hand_landmarks and vis_image is not None and output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to {output_path}")
        
        return all_landmarks_3d

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
    return detect_hand_landmarks_from_image(rgb_image, save_visualization, output_path)

def process_camera_feed(period=4.0, camera_id=0, save_visualization=False, output_dir="."):
    """
    Periodically capture frames from a camera and detect hand landmarks.
    
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
    
    print(f"Camera initialized. Capturing frames every {period} seconds.")
    print("Press 'q' to quit.")
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate output path for this frame if visualization is enabled
            output_path = None
            if save_visualization:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = f"{output_dir}/hand_landmarks_{timestamp}_{frame_count}.jpg"
            
            # Process the frame
            landmarks_3d = detect_hand_landmarks_from_image(rgb_frame, save_visualization, output_path)
            
            # Display frame with landmarks (always show the frame even if save_visualization is False)
            # We need to draw landmarks again since the original visualization might have been saved to a file
            display_frame = frame.copy()
            
            # Initialize MediaPipe Hand solution for visualization
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            # Process the frame again to get landmarks for visualization
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:
                
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
            
            # Display the frame count and hand detection status
            if landmarks_3d:
                status_text = f"Frame {frame_count}: {len(landmarks_3d)} hand(s) detected"
            else:
                status_text = f"Frame {frame_count}: No hands detected"
            
            cv2.putText(display_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(display_frame, "Press 'q' to quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Hand Tracking', display_frame)
            
            # Print landmark information
            if landmarks_3d:
                for i, hand_landmarks in enumerate(landmarks_3d):
                    print(f"Frame {frame_count} - Hand {i+1} landmarks:")
                    for j, (x, y, z) in enumerate(hand_landmarks):
                        print(f"  Landmark {j}: ({x:.2f}, {y:.2f}, {z:.2f})")
                    print()  # Empty line between hands
            else:
                print(f"Frame {frame_count}: No hands detected")
            
            # Increment frame count
            frame_count += 1
            
            # Wait for the specified period before capturing the next frame
            # Also check for 'q' key press during the wait period
            start_time = time.time()
            while (time.time() - start_time) < period:
                # Check for key press with a small timeout to keep it responsive
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt  # Use KeyboardInterrupt to break out of the loops
                # Sleep a bit to avoid hogging CPU
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Exiting...")
    
    finally:
        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()

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
        
        elif args.camera:
            # Process camera feed
            process_camera_feed(args.period, args.camera_id, args.visualize, args.output)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
