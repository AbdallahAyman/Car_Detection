import cv2
import numpy as np
#take video and define background subtraction
cap = cv2.VideoCapture('LB2.webm')
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
# define blob detector
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 200
params.maxThreshold = 255
params.minDistBetweenBlobs = 10
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByArea = True
params.minArea = 4000
params.maxArea = 400000
params.blobColor = 255
detector = cv2.SimpleBlobDetector_create(params)
while (cap.isOpened()):
#apply foreground detector
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
# define filter
    kernel = np.ones((5, 5), np.uint8)
    npn=[[]]
    if ret:
        #apply filters on masked frame
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # Detect blobs.
        keypoints = detector.detect(closing)
        #initialize some parameter that will be used later
        ny=[]
        dummy=[[]]
        if keypoints:
            nk = len(keypoints)

            for i in range(nk):
                #define bounding box parameters
                x1=keypoints[i].pt[0]
                y1=keypoints[i].pt[1]
                l=int(keypoints[i].size)
                x=int(x1-l/2)
                y=int(y1+l/2)
                if npn.__contains__([]):
                    npn.remove([])
                npn.append([int(x1), int(y1), l])
                #crop blobbox
                if x>0:
                    crop_img = frame[y - l:y, x:x + l]
                    # convert from bgr to hsv
                    image = hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

                    # define the list of boundaries
                    boundaries = [([15, 100, 100], [60, 255, 255])]  # yellow boundaries in hsv

                    lower = np.array(boundaries[0][0], dtype="uint8")
                    upper = np.array(boundaries[0][1], dtype="uint8")

                    # find the colors within the specified boundaries and apply
                    # the mask
                    mask = cv2.inRange(image, lower, upper)
                    output = cv2.bitwise_and(crop_img, crop_img, mask=mask)

                    # ratio of yellow in output
                    grey = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                    (thresh, bw) = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    non_zero = cv2.countNonZero(bw)
                    total_pixels = bw.size
                    non_zero = float(non_zero)
                    ratio = non_zero / total_pixels



                if ratio > 0.15:
                    npy = [[int(x1),int(y1),l]]
                    ny.append(i)
                    xs = int(x1 + l / 2)
                    ys = int(y1 + l / 2)
                    frame2=cv2.rectangle(frame, (xs-l, ys-l), (xs,ys), (20, 255, 255), 2)
                    cv2.imshow("Keypoints", frame2)


                else:
                    frame2=frame


            if len(ny) != 0:
                for j in range(len(ny)):
                    del npn[ny[j]]
                    for k in range(len(ny)):
                        ny[k] = ny[k] - 1



            for i in range(len(npn)):
                # Draw green rectangles around blobs
                if npn:
                    x1 = npn[i][0]
                    y1 = npn[i][1]
                    l = npn[i][2]
                    x = int(x1 + l / 2)
                    y = int(y1 + l / 2)
                    cv2.rectangle(frame2, (x-l,y-l), (x,y), (0, 255, 0), 2)

            frame2=cv2.putText(frame2, "%d" %len(npn), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 4)
            frame2 = cv2.putText(frame2, "%d"%len(ny), (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200),
                                 4)

            cv2.imshow("Keypoints", frame2)
            cv2.waitKey(50)
        else:
            frame2 = cv2.putText(frame, "0", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 4)
            frame2 = cv2.putText(frame, "0", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 4)
            cv2.imshow("Keypoints", frame2)
            cv2.waitKey(50)


    else:
        cap.release()
        break

cv2.destroyAllWindows()