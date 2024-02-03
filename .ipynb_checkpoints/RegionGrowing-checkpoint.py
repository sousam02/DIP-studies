import cv2
import numpy as np

def regionGrowing(anImage, aSeedSet, anInValue=255, tolerance=5):
    # Boolean array/matrix, same size as image
    # All the pixels are initialized to false
    visited_matrix = np.zeros_like(anImage, dtype=np.uint8)

    # List of points to visit
    point_list = aSeedSet

    while point_list:
        # Get a point from the list
        this_point = point_list.pop()
        x, y = this_point

        # Check if the point is inside the image bounds
        if 0 <= x < anImage.shape[1] and 0 <= y < anImage.shape[0]:
            pixel_value = anImage[y, x]

            # Visit the point
            visited_matrix[y, x] = anInValue

            # For each neighbor of this_point
            for j in range(y - 1, y + 2):
                for i in range(x - 1, x + 2):
                    # Check if the neighbor is inside the image bounds
                    if 0 <= i < anImage.shape[1] and 0 <= j < anImage.shape[0]:
                        neighbour_value = anImage[j, i]
                        neighbour_visited = visited_matrix[j, i]

                        if (
                            not neighbour_visited
                            and abs(neighbour_value - pixel_value) <= (tolerance / 100.0 * 255.0)
                        ):
                            point_list.append((i, j))

    return visited_matrix

# Global variables to store the seed coordinates
seed = (-1, -1)

def mouse_callback(event, x, y, flags, param):
    global seed
    
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (x, y)
        print(f"Seed set at ({x}, {y})")

# Read your image
image = cv2.imread("assets/CT.png", 0)

# Create a window and set the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)
while True:
    # Display the image
    cv2.imshow("Image", image)

    # Break the loop if the 'esc' key is pressed or seed is set
    if cv2.waitKey(1) & 0xFF == 27 or seed != (-1, -1):
        break
    
segmented_image = regionGrowing(image, [seed], tolerance=20)

# Display the original and segmented images
cv2.imshow("Original Image", image)
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()