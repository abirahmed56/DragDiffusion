import cv2
import numpy as np
import os

# -------------------- Settings --------------------
image_path = "abir.jpeg"   # input image
mask_path  = "mask.png"    # where the mask will be saved
mask_thickness = 15        # thickness of brush

# -------------------- Load Image --------------------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read {image_path}")

# Initialize empty mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# -------------------- Mouse Callback --------------------
drawing = False
ix, iy = -1, -1

def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(mask, (ix, iy), (x, y), color=255, thickness=mask_thickness)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(mask, (ix, iy), (x, y), color=255, thickness=mask_thickness)

cv2.namedWindow("Draw Mask (Press 's' to save, 'q' to quit)")
cv2.setMouseCallback("Draw Mask (Press 's' to save, 'q' to quit)", draw_mask)

# -------------------- Draw Loop --------------------
while True:
    overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow("Draw Mask (Press 's' to save, 'q' to quit)", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved to {mask_path}")
        break
    elif key == ord('q'):
        print("Mask drawing canceled.")
        break

cv2.destroyAllWindows()
