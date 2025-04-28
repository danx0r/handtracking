import cv2
import mediapipe as mp
import numpy as np
import argparse

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

def main():
    parser = argparse.ArgumentParser(description='Detect 3D hand landmarks from an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--visualize', '-v', action='store_true', help='Save visualization of landmarks')
    parser.add_argument('--output', '-o', type=str, default='hand_landmarks.jpg', help='Path to save visualization')
    
    args = parser.parse_args()
    
    try:
        # Detect hand landmarks from file
        landmarks_3d = detect_hand_landmarks_from_file(args.image_path, args.visualize, args.output)
        
        if not landmarks_3d:
            print("No hands detected in the image.")
            return
        
        # Print the 3D landmarks for each detected hand
        for i, hand_landmarks in enumerate(landmarks_3d):
            print(f"Hand {i+1} landmarks:")
            for j, (x, y, z) in enumerate(hand_landmarks):
                print(f"  Landmark {j}: ({x:.2f}, {y:.2f}, {z:.2f})")
            print()  # Empty line between hands
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
