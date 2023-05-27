	from scipy.spatial import distance as dist
	import numpy as np
	import mahotas
	import cv2
	import imutils
	def describe_shapes(image):
    		shapeFeatures = []
  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]
  
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.erode(thresh, None, iterations=2)
  
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
  
    for c in cnts:
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        (x, y, w, h) = cv2.boundingRect(c)
        roi = mask[y:y + h, x:x + w]
  
        features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
        shapeFeatures.append(features)
  
    return (cnts, shapeFeatures)

refImage = cv2.imread("pokemon_red.png")
(_, gameFeatures) = describe_shapes(refImage)
  
shapesImage = cv2.imread("shapess.png")
(cnts, shapeFeatures) = describe_shapes(shapesImage)
  
D = dist.cdist(gameFeatures, shapeFeatures)
i = np.argmin(D)
 
for (j, c) in enumerate(cnts):
    if i != j:
        box = cv2.minAreaRect(c)
        box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
        cv2.drawContours(shapesImage, [box], -1, (0, 0, 255), 2)


# In[ ]:




