from config import denoise_config as config
from denoising.helpers import blur_and_threshold
from imutils import paths
import pickle
import random
import cv2

model = pickle.loads(open(config.model_path, 'rb').read())

imagePaths = list(paths.list_images(config.test_path))
random.shuffle(imagePaths)
imagePaths = imagePaths[:5]

for path in imagePaths:
    print("----Processing {}----".format(path))
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original = image.copy()

    image = cv2.copyMakeBorder(image, 2,2,2,2, cv2.BORDER_REPLICATE)
    image = blur_and_threshold(image)

    roiFeatures = []

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            roi = image[y:y+5, x:x+5]
            (rH, rW) = roi.shape[:2]

            if rH!=5 or rW!=5:
                continue

            features = roi.flatten()
            roiFeatures.append(features)

    pixels = model.predict(roiFeatures)

    pixels = pixels.reshape(original.shape)
    output = (pixels * 255).astype('uint8')

    cv2.imshow("Original", original)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
