# import the necessary packages
import cv2

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.053 * peri, True)
		area = cv2.contourArea(c)

		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square"

		# if the shape is a pentagon, it will have 5 vertices

		# otherwise, we assume the shape is a circle
		elif area < 45:
			shape = "5 cm circle"

		else:
			shape = "10 cm circle"

		# return the name of the shape
		return shape
