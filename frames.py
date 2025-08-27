import cv2
import os

# Path to the video file
video_path = 'test_videos/temp-07012025193856-0000.avi'  # Change this to your video file
output_folder = 'frames_output_3'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

# Read and save frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop when video ends

    # Save frame as JPEG
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    print(f'Saved {frame_filename}')

    frame_count += 1

# Release the video capture object
cap.release()
print("Done saving frames.")
