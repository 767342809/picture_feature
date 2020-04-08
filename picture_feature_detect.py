import ssl
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
import numpy as np

from skimage.io import imread

ssl._create_default_https_context = ssl._create_unverified_context


def look_hog_picture(img_path, is_resize):
    fd, hog_image, image = detect_hog_feature(img_path, is_resize)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    return fd


def detect_hog_feature(img_path, is_resize: bool=True):
    image = imread(img_path, plugin='pil')
    if is_resize:
        image = resize(image, (224, 224), preserve_range=True, anti_aliasing=False, clip=False)

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(3, 3), visualize=True, multichannel=True)
    return fd, hog_image, image


if __name__ == "__main__":
    img_file = "https://wx1.sinaimg.cn/orj1080/53db7999gy1gd0cxmhljwj215o0rs4qp.jpg"
    img_file2 = "./tfrecord/resize.jpg"
    img_file3 = "./test_data/puzzle.jpeg"
    fd1, _, _ = detect_hog_feature(img_file)
    fd2, _, _ = detect_hog_feature(img_file2)
    fd3, _, _ = detect_hog_feature(img_file3)
    all = np.array([fd1.tolist(), fd2.tolist(), fd3])
    print(all)
    print(np.corrcoef(all))


    p = all
    q = fd2
    BC = np.sum(np.sqrt(p * q), axis=1)
    print("BC: ", BC)
    men = np.mean(p, axis=1) * np.mean(q) * np.power(q.shape[0], 2)
    print("men: ", men)
    s = 1 - (BC / np.sqrt(men))
    d = np.sqrt(s)
    print(d)
