import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your pikachu.png (ensure it has transparent background)
pikachu_img = cv2.imread("pikachu.png", cv2.IMREAD_UNCHANGED)

# Check if image loaded properly
if pikachu_img is None:
    print("Error: Could not load pikachu.png! Check file path.")
    exit()

def overlay_pikachu(frame, face_x, face_y, face_w, face_h):
    # Calculate position to center Pikachu on face
    pikachu_w = int(face_w * 1.5)  # Make Pikachu slightly wider than face
    pikachu_h = int(face_h * 1.2)  # Slightly taller than face
    
    # Position adjustment to center on face
    x = face_x - int((pikachu_w - face_w)/2)
    y = face_y - int(pikachu_h * 0.3)  # Move up to cover forehead
    
    # Resize Pikachu while maintaining aspect ratio
    aspect_ratio = pikachu_img.shape[1] / pikachu_img.shape[0]
    pikachu_h = int(pikachu_w / aspect_ratio)
    resized_pikachu = cv2.resize(pikachu_img, (pikachu_w, pikachu_h))
    
    # Overlay with transparency
    alpha = resized_pikachu[:, :, 3] / 255.0
    for c in range(3):
        if y + pikachu_h > frame.shape[0] or x + pikachu_w > frame.shape[1]:
            continue  # Skip if out of bounds
        try:
            frame[y:y+pikachu_h, x:x+pikachu_w, c] = \
                (1 - alpha) * frame[y:y+pikachu_h, x:x+pikachu_w, c] + \
                alpha * resized_pikachu[:, :, c]
        except:
            pass
    
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = overlay_pikachu(frame, x, y, w, h)
        # Add Pikachu text
        cv2.putText(frame, "PIKACHU!", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Pikachu Face Filter (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()