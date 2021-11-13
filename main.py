import cv2 as cv
import numpy as np

def show_image(title,image):
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def preprocess_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_m_blur = cv.medianBlur(image, 5)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.erode(thresh, kernel)

    show_image("median blurred", image_m_blur)
    show_image("gaussian blurred", image_g_blur)
    show_image("sharpened", image_sharpened)
    show_image("threshold of blur", thresh)

    edges = cv.Canny(thresh, 150, 400)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 500
    height = 500

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(image_copy, tuple(top_left), 4, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 4, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 4, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 4, (0, 0, 255), -1)
    show_image("detected corners", image_copy)

    return top_left, top_right, bottom_left, bottom_right


img = cv.imread("antrenare/clasic/01.jpg")
img = cv.resize(img,(0,0),fx=0.2,fy=0.2)
result=preprocess_image(img)


points = np.float32([[result[0][0], result[0][1]], [result[1][0], result[1][1]], [result[2][0], result[2][1]], [result[3][0], result[3][1]]])
pts2 = np.float32([[0, 0], [499, 0], [0, 499], [499, 499]])

matrix = cv.getPerspectiveTransform(points, pts2)
res = cv.warpPerspective(img, matrix, (500, 500))

show_image("res", res)

patch_dim = 500//9

for i in range(9):
    for j in range(9):
        patch = res[i*patch_dim+10:(i+1)*patch_dim-10, j*patch_dim+10:(j+1)*patch_dim-10]
        show_image("patch", patch)

        gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        max1 = 0
        for k in contours:
            if cv.contourArea(k) > max1:
                max1 = cv.contourArea(k)

        print(max1)