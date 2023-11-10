from config import denoise_config as config
from denoising.helpers import blur_and_threshold
from imutils import paths
import progressbar
import random
import cv2

trainPaths = sorted(list(paths.list_images(config.train_path)))
cleanedPaths = sorted(list(paths.list_images(config.cleaned_path)))

widgets = ["Creating Features: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trainPaths),
	widgets=widgets).start()

# Zipping the training paths together and opening a CSV file to write all the extracted features in it
imagePaths = list(zip(trainPaths, cleanedPaths))
csv = open(config.features_path, 'w')

for (i, (trainPath, cleanedPath)) in enumerate(imagePaths):

    # Loading the train (noisy) and corresponding clean images and converting them to grayscale
    trainImage = cv2.imread(trainPath)
    cleanImage = cv2.imread(cleanedPath)
    trainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
    cleanImage = cv2.cvtColor(cleanImage, cv2.COLOR_BGR2GRAY)

    # applying 2*2 padding to both images, replicating the pixels along the border so that output image is not smaller in size
    trainImage = cv2.copyMakeBorder(trainImage, 2,2,2,2, cv2.BORDER_REPLICATE)
    cleanImage = cv2.copyMakeBorder(cleanImage, 2,2,2,2, cv2.BORDER_REPLICATE)

    # Blurring and thresholding the images
    trainImage = blur_and_threshold(trainImage)

    # Scaling the pixel intensities in the clean images from [0,255] to [0,1] (Since the train images (noisy) are already in range [0,1])
    cleanImage = cleanImage.astype('float') / 255.0

    # Sliding a 5*5 window across the images
    for y in range(trainImage.shape[0]):
        for x in range(trainImage.shape[1]):
	    # Extracting the window ROI (Region Of Interests) for train and clean images, along with their spatial dimensions (height and width)
            trainROI = trainImage[y:y+5, x:x+5]
            cleanROI = cleanImage[y:y+5, x:x+5]
            (rH, rW) = trainROI.shape[:2]

	    # Checking if the ROIs are 5*5, if not we can throw them out (because if they are not 5*5 that mean they are the windows at borders, which we don't care about)
            if rH!=5 or rW!=5:
                continue

	    # Features are flattened 5*5=25 raw pixels from the noisy ROIs and the target prediction will be the center pixel in the 5*5 window ([2,2])
            features = trainROI.flatten()
            target = cleanROI[2,2]

            if random.random() <= config.sample_prob:
                features = [str(x) for x in features]
                row = [str(target)] + features
                row = ",".join(row)
                csv.write("{}\n".format(row))

    pbar.update(i)

pbar.finish()
csv.close()

    
