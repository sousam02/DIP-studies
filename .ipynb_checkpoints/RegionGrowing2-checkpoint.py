import cv2
import numpy as np

def mouse_callback(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_set.append((x, y))
        p_image = userdata
        color = (0, 0, 255)
        cv2.circle(p_image, (x, y), 4, color, cv2.FILLED)
        cv2.imshow("CT slice", p_image)

def regionGrowing(anImage, aSeedSet, anInValue=255, tolerance=5):
    visited_matrix = np.zeros_like(anImage, dtype=np.uint8)

    point_list = aSeedSet

    while point_list:
        this_point = point_list.pop()
        x, y = this_point

        if 0 <= x < anImage.shape[1] and 0 <= y < anImage.shape[0]:
            pixel_value = anImage[y, x]

            visited_matrix[y, x] = anInValue

            for j in range(y - 1, y + 2):
                if 0 <= j < anImage.shape[0]:
                    for i in range(x - 1, x + 2):
                        if 0 <= i < anImage.shape[1]:
                            neighbour_value = anImage[j, i]
                            neighbour_visited = visited_matrix[j, i]

                            if not neighbour_visited and abs(int(neighbour_value) - int(pixel_value)) <= (tolerance / 100.0 * 255.0):
                                point_list.append((i, j))

    return visited_matrix

ct_slice = cv2.imread("assets/CT.png", 0);
seed_set = []
colour_ct_slice = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)

cv2.namedWindow("CT slice", cv2.WINDOW_AUTOSIZE)
cv2.imshow("CT slice", colour_ct_slice)
cv2.setMouseCallback("CT slice", mouse_callback, colour_ct_slice)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Seed set:", seed_set)

segmented_image = regionGrowing(ct_slice, seed_set, 255, 2);

cv2.namedWindow("Segmentation", cv2.WINDOW_AUTOSIZE)  # Create a window
cv2.imshow("Segmentation", segmented_image)  # Show the segmented image inside the created window

cv2.waitKey(0)  # Wait for any keystroke in the window

cv2.destroyAllWindows()