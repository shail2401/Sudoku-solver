import numpy as np
from numpy.lib.function_base import append
from skimage.segmentation import clear_border
import cv2

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    # Bottom-Right coordinate has largest sum and Top-Left has the lowest
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    s = np.diff(pts, axis=1)
    # Top right has the smallest difference and Bottom Left has the highest difference
    rect[1] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((bl[0]-br[0])**2 - (bl[1]-br[1])**2))
    widthB = np.sqrt(((tl[0]-tr[0])**2 - (tl[1]-tr[1])**2))
    maxwidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinate
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxheight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
        [0, 0],
        [maxwidth-1, 0],
        [maxwidth-1, maxheight-1],
        [0, maxheight-1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxwidth, maxheight))

    return warped

def grab_contours(contours):
    if len(contours) == 2:
        cnts = contours[0]
    elif len(contours) == 3:
        cnts = contours[1]
    return cnts


def find_puzzle(image, debug=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Puzzle Thresh" ,thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # find contours in the thresholded image and sort them by size in
	# descending order
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzlecnt = 0

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, .02*peri, True)

        if len(approx) == 4:
            puzzlecnt = approx
            break

        # if the puzzle contour is empty then our script could not find
	    # the outline of the Sudoku puzzle so raise an error

        if not puzzlecnt:
            raise Exception('No Sudoku puzzle found. Try checking functions.')
        
        # check to see if we are visualizing the outline of the detected
	    # Sudoku puzzle
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzlecnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
    warped = four_point_transform(gray, puzzlecnt.reshape(4, 2))
    puzzle = four_point_transform(image, puzzlecnt.reshape(4, 2))

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return puzzle, warped

def extract_digits(cell, debug=True):
    # apply automatic thresholding to the cell and then clear
    # any connected border that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # find contours in the thresholded cells
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    # if no contours found that is a empty cell
    if not len(cnts):
        return None
    
    # otherwise, find the largest contour in the cell and create a
	# mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)

    # if no contours were found then this is an empty cell
    if len(cnts) == 0:
        return None

    # otherwise find the largest contour in the cell and create a mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total area of image
    (h, w) = thresh.shape
    percentfilled = cv2.countNonZero(mask)/float(w*h)

    # if less than 3% of the mask is filled then noise and can safely ignore the contour
    if percentfilled < 0.03:
        return None

    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow('Digit', digit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return digit
    