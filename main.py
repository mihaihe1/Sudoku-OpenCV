import cv2 as cv
import numpy as np
from sklearn import svm
from collections import deque
from tensorflow.keras.datasets import mnist


# train_images = np.loadtxt('data/train_images.txt')
# train_labels = np.loadtxt('data/train_labels.txt', 'int32')
#
# test_images = np.loadtxt('data/test_images.txt')
# test_labels = np.loadtxt('data/test_labels.txt', 'int32')
# imm = np.reshape(test_images[2], (28, 28))
# clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
# clf.fit(train_images, train_labels)
#
# print(clf.score(test_images, test_labels))


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

    # show_image("median blurred", image_m_blur)
    # show_image("gaussian blurred", image_g_blur)
    # show_image("sharpened", image_sharpened)
    # show_image("threshold of blur", thresh)

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
    # show_image("detected corners", image_copy)

    return top_left, top_right, bottom_left, bottom_right


def task1():
    for cnt in range(1, 21):
        if cnt < 10:
            img = cv.imread(f"antrenare/clasic/0{cnt}.jpg")
            f = open(f'predicted/clasic/0{cnt}_predicted.txt', 'w')
        else:
            img = cv.imread(f"antrenare/clasic/{cnt}.jpg")
            f = open(f'predicted/clasic/{cnt}_predicted.txt', 'w')

        img = cv.resize(img,(0,0),fx=0.2,fy=0.2)
        result=preprocess_image(img)

        points = np.float32([[result[0][0], result[0][1]], [result[1][0], result[1][1]], [result[2][0], result[2][1]], [result[3][0], result[3][1]]])
        pts2 = np.float32([[0, 0], [499, 0], [0, 499], [499, 499]])

        matrix = cv.getPerspectiveTransform(points, pts2)
        res = cv.warpPerspective(img, matrix, (500, 500))

        # show_image("res", res)

        patch_dim = 500//9
        occupied_threshold = 50

        for i in range(9):
            for j in range(9):
                p1 = res[i*patch_dim:(i+1)*patch_dim, j*patch_dim:(j+1)*patch_dim]
                # show_image("p1", p1)
                patch = res[i*patch_dim+15:(i+1)*patch_dim-10, j*patch_dim+15:(j+1)*patch_dim-10]

                grid = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                gr = cv.resize(grid, (28, 28), interpolation=cv.INTER_AREA)
                for x in range(28):
                    for y in range(28):
                        if gr[x][y] > 127:
                            gr[x][y] = 255
                # show_image("g", gr)
                g = gr.reshape(28, 28, 1)
                g = g.reshape(1, 28, 28, 1)

                print(g)
                # grid = cv.bitwise(cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 1))
                # resized_patch1 = cv.resize(grid, (28, 28), interpolation=cv.INTER_AREA)
                # show_image("grid", grid)
                # print("size", patch.shape )
                resized_patch = cv.resize(patch, (28, 28), interpolation=cv.INTER_AREA)
                out = cv.normalize(resized_patch.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
                gr = cv.cvtColor(resized_patch, cv.COLOR_BGR2GRAY)
                gr = gr.flatten()
                ok = 0
                for k in range(len(gr)):
                    if gr[k] <= 100:
                        ok = 1
                        gr[k] = 255
                    else:
                        gr[k] = 0
                # gr = resized_patch1.reshape(1, -1)
                # if ok == 1:occupied_threshold=" ")
                # gr = np.reshape(gr, (28, 28))
                # show_image("patch", gr)
                # print(clf.predict(gr))

                gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                blur = cv.GaussianBlur(gray, (5, 5), 0)
                show_image("i", blur)
                thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
                show_image("i", thresh)
                contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                max1 = 0
                for k in contours:
                    if cv.contourArea(k) > max1:
                        max1 = cv.contourArea(k)
                if max1 <= occupied_threshold:
                    f.write("o")
                else:
                    f.write("x")

                # print(max1)
            if i != 8:
                f.write("\n")


def task2():
    for cnt in range(1, 41):
        if cnt < 10:
            img = cv.imread(f"antrenare/jigsaw/0{cnt}.jpg")
            f = open(f'predicted/jigsaw/0{cnt}_predicted.txt', 'w')
        else:
            img = cv.imread(f"antrenare/jigsaw/{cnt}.jpg")
            f = open(f'predicted/jigsaw/{cnt}_predicted.txt', 'w')

        img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
        result = preprocess_image(img)

        points = np.float32([[result[0][0], result[0][1]], [result[1][0], result[1][1]], [result[2][0], result[2][1]],
                             [result[3][0], result[3][1]]])
        pts2 = np.float32([[0, 0], [499, 0], [0, 499], [499, 499]])

        matrix = cv.getPerspectiveTransform(points, pts2)
        res = cv.warpPerspective(img, matrix, (500, 500))

        # show_image("res", res)

        patch_dim = 500 // 9
        occupied_threshhold = 50
        color_matrix = [['x' for x in range(9)] for j in range(9)]
        occupied_matrix = [['' for x in range(9)] for j in range(9)]
        cnt_color = 0
        cnt_alb_negru = 0

        for i in range(9):
            patch = res[i * patch_dim:(i + 1) * patch_dim, 0: patch_dim]
            a1 = np.mean(patch, axis=tuple(range(2)))
            if a1.max(axis=0)-a1.min(axis=0) <= 7:
                cnt_alb_negru += 1
            else:
                cnt_color += 1

        if max(cnt_color, cnt_alb_negru) == cnt_color:
            for i in range(9):
                for j in range(9):
                    patch = res[i * patch_dim + 10:(i + 1) * patch_dim - 10, j * patch_dim + 10:(j + 1) * patch_dim - 10]
                    patch_margins = res[i * patch_dim:(i + 1) * patch_dim, j * patch_dim:(j + 1) * patch_dim]
                    a1 = np.mean(patch_margins, axis=tuple(range(2)))
                    # print(np.mean(patch_margins, axis=tuple(range(2))))
                    # print(a1.max(axis=0)-a1.min(axis=0))
                    # show_image("patch", patch)
                    # print(np.mean(patch_  margins, axis=tuple(range(2))))
                    median = cv.medianBlur(patch, 5)
                    blur_t = cv.GaussianBlur(patch, (5, 5), 0)
                    # print(np.mean(blur_t, axis=tuple(range(2))))
                    mean = np.mean(patch, axis=tuple(range(2)))
                    print(np.mean(patch, axis=tuple(range(2))))
                    if mean[0] == max(mean):
                        color_matrix[i][j] = 'B'
                    elif mean[2] == max(mean) and abs(mean[2]-mean[1]) > 5:
                        color_matrix[i][j] = 'R'
                    else:
                        color_matrix[i][j] = 'Y'
                    # print(color_matrix[i][j], end="")

                    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                    blur = cv.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv.n (blur, 255, 1, 1, 11, 2)
                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                    max1 = 0
                    for k in contours:
                        if cv.contourArea(k) > max1:
                            max1 = cv.contourArea(k)
                    if max1 <= occupied_threshhold:
                        occupied_matrix[i][j] = 'o'
                        # print("o", end="")
                    else:
                        occupied_matrix[i][j] = 'x'
                        # print("x", end="")

                    # print(max1)
                print()

            numbered_matrix = [[0 for x in range(9)] for j in range(9)]
            viz = [[0 for x in range(9)] for j in range(9)]
            directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            cnt = 0
            for i in range(9):
                for j in range(9):
                    if viz[i][j] == 0:
                        cnt += 1
                        q = deque([(i, j)])
                        color = color_matrix[i][j]
                        while len(q):
                            coord = q.popleft()
                            x = coord[0]
                            y = coord[1]
                            viz[x][y] = 1
                            numbered_matrix[x][y] = cnt
                            for dx, dy in directions:
                                if 0 <= x+dx <= 8 and 0 <= y+dy <= 8 and viz[x+dx][y+dy] == 0 and color_matrix[x+dx][y+dy] == color:
                                    q.append((x+dx, y+dy))
            for i in range(9):
                for j in range(9):
                    f.write(str(numbered_matrix[i][j]))
                    f.write(str(occupied_matrix[i][j]))
                if i != 8:
                    f.write("\n")
        else:
            right_blocked = [[0 for x in range(9)] for j in range(9)]
            left_blocked = [[0 for x in range(9)] for j in range(9)]
            top_blocked = [[0 for x in range(9)] for j in range(9)]
            bottom_blocked = [[0 for x in range(9)] for j in range(9)]

            treshold = 120
            for i in range(9):
                for j in range(9):
                    patch_margins = res[i * patch_dim:(i + 1) * patch_dim, j * patch_dim:(j + 1) * patch_dim]
                    # print(np.mean(patch_margins, axis=tuple(range(2))))
                    a1 = np.mean(patch_margins, axis=tuple(range(2)))
                    print(np.mean(patch_margins, axis=tuple(range(2))))
                    print(a1.max(axis=0) - a1.min(axis=0))
                    if i == 0:
                        top_blocked[i][j] = 1
                    else:
                        top_margin = patch_margins[0:5, :]
                        gray_top = cv.cvtColor(top_margin, cv.COLOR_BGR2GRAY)
                        if np.mean(gray_top) < 130:
                            top_blocked[i][j] = 1
                            bottom_blocked[i - 1][j] = 1

                    if j == 0:
                        left_blocked[i][j] = 1
                    else:
                        left_margin = patch_margins[:, 0:5]
                        gray_left = cv.cvtColor(left_margin, cv.COLOR_BGR2GRAY)
                        # if i == 5 and j == 3:
                        #     show_image("g", gray_left)
                        if np.mean(gray_left) < 130:
                            left_blocked[i][j] = 1
                            right_blocked[i][j - 1] = 1

                    if j == 8:
                        right_blocked[i][j] = 1
                    if i == 8:
                        bottom_blocked[i][j] = 1
                    # show_image("top margin", top_margin)
                    # print(np.mean(gray_top))
                    # left_margin = patch[:, 0:5]
                    # gray_left = cv.cvtColor(left_margin, cv.COLOR_BGR2GRAY)
                    # show_image("patch", gray)
                    # mean = np.mean(left_margin, axis=tuple(range(2)))
                    # print(np.mean(patch, axis=tuple(range(2))))
                    # print("normal", mean)
                    # print("gray", np.mean(gray))
                    patch = res[i * patch_dim + 10:(i + 1) * patch_dim - 10,
                            j * patch_dim + 10:(j + 1) * patch_dim - 10]
                    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                    blur = cv.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                    max1 = 0
                    for k in contours:
                        if cv.contourArea(k) > max1:
                            max1 = cv.contourArea(k)
                    if max1 <= occupied_threshhold:
                        occupied_matrix[i][j] = 'o'
                        # print("o", end="")
                    else:
                        occupied_matrix[i][j] = 'x'
                        # print("x", end="")

                    # print(max1)
                # print()
            print("TOP")
            for i in range(9):
                for j in range(9):
                    print(top_blocked[i][j], end="")
                print()
            print("RIGHT")
            for i in range(9):
                for j in range(9):
                    print(right_blocked[i][j], end="")
                print()
            print("BOTTOM")
            for i in range(9):
                for j in range(9):
                    print(bottom_blocked[i][j], end="")
                print()
            print("LEFT")
            for i in range(9):
                for j in range(9):
                    print(left_blocked[i][j], end="")
                print()

            numbered_matrix = [[0 for x in range(9)] for j in range(9)]
            viz = [[0 for x in range(9)] for j in range(9)]
            # directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            cnt = 0
            for i in range(9):
                for j in range(9):
                    if viz[i][j] == 0:
                        cnt += 1
                        q = deque([(i, j)])
                        # color = color_matrix[i][j]
                        while len(q):
                            coord = q.popleft()
                            x = coord[0]
                            y = coord[1]
                            viz[x][y] = 1
                            numbered_matrix[x][y] = cnt

                            if top_blocked[x][y] == 0 and viz[x - 1][y] == 0:
                                q.append((x - 1, y))
                            if right_blocked[x][y] == 0 and viz[x][y + 1] == 0:
                                q.append((x, y + 1))
                            if bottom_blocked[x][y] == 0 and viz[x + 1][y] == 0:
                                q.append((x + 1, y))
                            if left_blocked[x][y] == 0 and viz[x][y - 1] == 0:
                                q.append((x, y - 1))

            for i in range(9):
                for j in range(9):
                    f.write(str(numbered_matrix[i][j]))
                    f.write(str(occupied_matrix[i][j]))
                if i != 8:
                    f.write("\n")

def test_margins():
    img = cv.imread("antrenare/jigsaw/36.jpg")
    img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    result = preprocess_image(img)

    points = np.float32([[result[0][0], result[0][1]], [result[1][0], result[1][1]], [result[2][0], result[2][1]],
                         [result[3][0], result[3][1]]])
    pts2 = np.float32([[0, 0], [499, 0], [0, 499], [499, 499]])

    matrix = cv.getPerspectiveTransform(points, pts2)
    res = cv.warpPerspective(img, matrix, (500, 500))

    show_image("res", res)

    patch_dim = 500 // 9
    occupied_threshhold = 50
    color_matrix = [['x' for x in range(9)] for j in range(9)]
    occupied_matrix = [['' for x in range(9)] for j in range(9)]

    right_blocked = [[0 for x in range(9)] for j in range(9)]
    left_blocked = [[0 for x in range(9)] for j in range(9)]
    top_blocked = [[0 for x in range(9)] for j in range(9)]
    bottom_blocked = [[0 for x in range(9)] for j in range(9)]

    treshold = 120
    for i in range(9):
        for j in range(9):
            patch_margins = res[i * patch_dim:(i + 1) * patch_dim, j * patch_dim:(j + 1) * patch_dim]
            # print(np.mean(patch_margins, axis=tuple(range(2))))
            a1 = np.mean(patch_margins, axis=tuple(range(2)))
            print(np.mean(patch_margins, axis=tuple(range(2))))
            print(a1.max(axis=0)-a1.min(axis=0))
            if i == 0:
                top_blocked[i][j] = 1
            else:
                top_margin = patch_margins[0:5, :]
                gray_top = cv.cvtColor(top_margin, cv.COLOR_BGR2GRAY)
                if np.mean(gray_top) < 120:
                    top_blocked[i][j] = 1
                    bottom_blocked[i-1][j] = 1

            if j == 0:
                left_blocked[i][j] = 1
            else:
                left_margin = patch_margins[:, 0:5]
                gray_left = cv.cvtColor(left_margin, cv.COLOR_BGR2GRAY)
                if np.mean(gray_left) < 120:
                    left_blocked[i][j] = 1
                    right_blocked[i][j-1] = 1

            if j == 8:
                right_blocked[i][j] = 1
            if i == 8:
                bottom_blocked[i][j] = 1
            # show_image("top margin", top_margin)
            # print(np.mean(gray_top))
            # left_margin = patch[:, 0:5]
            # gray_left = cv.cvtColor(left_margin, cv.COLOR_BGR2GRAY)
            # show_image("patch", gray)
            # mean = np.mean(left_margin, axis=tuple(range(2)))
            # print(np.mean(patch, axis=tuple(range(2))))
            # print("normal", mean)
            # print("gray", np.mean(gray))
            patch = res[i * patch_dim + 10:(i + 1) * patch_dim - 10, j * patch_dim + 10:(j + 1) * patch_dim - 10]
            gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (5, 5), 0)
            thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            max1 = 0
            for k in contours:
                if cv.contourArea(k) > max1:
                    max1 = cv.contourArea(k)
            if max1 <= occupied_threshhold:
                occupied_matrix[i][j] = 'o'
                # print("o", end="")
            else:
                occupied_matrix[i][j] = 'x'
                # print("x", end="")

            # print(max1)
        # print()

    numbered_matrix = [[0 for x in range(9)] for j in range(9)]
    viz = [[0 for x in range(9)] for j in range(9)]
    # directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    cnt = 0
    for i in range(9):
        for j in range(9):
            if viz[i][j] == 0:
                cnt += 1
                q = deque([(i, j)])
                # color = color_matrix[i][j]
                while len(q):
                    coord = q.popleft()
                    x = coord[0]
                    y = coord[1]
                    viz[x][y] = 1
                    numbered_matrix[x][y] = cnt

                    if top_blocked[x][y] == 0 and viz[x-1][y] == 0:
                        q.append((x-1, y))
                    if right_blocked[x][y] == 0 and viz[x][y+1] == 0:
                        q.append((x, y+1))
                    if bottom_blocked[x][y] == 0 and viz[x+1][y] == 0:
                        q.append((x+1, y))
                    if left_blocked[x][y] == 0 and viz[x][y-1] == 0:
                        q.append((x, y-1))

    for i in range(9):
        for j in range(9):
            print(numbered_matrix[i][j], end="")
            print(occupied_matrix[i][j], end="")
        print()

    # print("TOP")
    # for i in range(9):
    #     for j in range(9):
    #         print(top_blocked[i][j], end="")
    #     print()
    # print("RIGHT")
    # for i in range(9):
    #     for j in range(9):
    #         print(right_blocked[i][j], end="")
    #     print()
    # print("BOTTOM")
    # for i in range(9):
    #     for j in range(9):
    #         print(bottom_blocked[i][j], end="")
    #     print()
    # print("LEFT")
    # for i in range(9):
    #     for j in range(9):
    #         print(left_blocked[i][j], end="")
    #     print()

task1()
# image = np.reshape(test_images[2], (28, 28))
# show_image("img",image)