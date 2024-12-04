import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read the video or image sequence
cap = cv2.VideoCapture('/content/bird.jpg')  # Replace with your video path, or use 0 for webcam

# Initialize background subtractor (MOG2 method)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Step 1: Convert frame to grayscale for boundary detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Apply Canny edge detector for boundary detection
    edges = cv2.Canny(blurred, 50, 150)  # You can adjust the thresholds
    
    # Step 4: Apply background subtraction to extract the moving objects (foreground)
    fgmask = fgbg.apply(frame)  # Apply the background subtractor
    
    # Optional: Perform morphological operations to clean the foreground mask
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Close small holes
    
    # Optional: You can also perform additional filtering to refine the boundary detection results
    
    # Step 5: Combine the boundary detection and foreground mask
    result = cv2.bitwise_and(frame, frame, mask=fgmask)  # Mask the foreground
    boundary_result = cv2.bitwise_and(edges, fgmask)  # Combine edges with foreground mask
    
    # Show the results
    cv2_imshow(frame)           # Original frame
    cv2_imshow(edges)           # Boundary detection result
    cv2_imshow(fgmask)          # Foreground mask after background subtraction
    cv2_imshow(result)          # Final result with foreground objects highlighted
    
    # Break the loop if 'q' is pressed (useful for video streams)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close any open windows
cap.release()
cv2.destroyAllWindows()![bird](https://github.com/user-attachments/assets/6cfc8ba7-9ed8-4a57-962e-fc754023e161)
