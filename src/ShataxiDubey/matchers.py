import cv2
import numpy as np 
import random

class matchers:
	def __init__(self):
		self.sift = cv2.SIFT_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def match(self, i1, i2, direction=None):
		imageSet1 = self.getSIFTFeatures(i1)
		imageSet2 = self.getSIFTFeatures(i2)

		bf = cv2.BFMatcher()
		rawMatches = bf.knnMatch(imageSet2['des'], imageSet1['des'], 2)
		# print(f'rawmatches {type(rawMatches)}, {len(rawMatches[0])}, {rawMatches[0]}')
		matches = []
		for m,n in rawMatches:
			if m.distance < 0.75*n.distance:
				matches.append((m.trainIdx, m.queryIdx))

		print("Direction : ", direction)

		if len(matches) > 4:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']

			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in matches]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in matches]
				)
			
			H = self.homography_matrix_with_ransac(matchedPointsCurrent, matchedPointsPrev)
			return H
		return None

	def homography_matrix_with_ransac(self, query_pts, train_pts):
		threshold = 10 
		best_inliers = []
		best_H = None
		pts_arr = list(zip(query_pts, train_pts))
		for _ in range(500):
			
			pts = random.choices(pts_arr, k = 4)
			# print(f'four random points {pts}')
			src_pts = np.array([pt[0] for pt in pts])  
			dst_pts = np.array([pt[1] for pt in pts])
			H = self.find_homography_matrix(src_pts, dst_pts)
			inliers = []
			for pt_query, pt_train in pts_arr:
				estimated_pt = H @ np.array([pt_query[0], pt_query[1], 1]).T  # H is applied on right image
				estimated_pt = estimated_pt / estimated_pt[-1]
				distance = np.linalg.norm(estimated_pt[:2] -  np.array(pt_train)) # find difference with left coordiante
				if distance < threshold:
					inliers.append(((pt_query, pt_train)))
			
			if len(inliers) > len(best_inliers):
				best_inliers, best_H = inliers, H
		
		return best_H     

	def find_homography_matrix(self, query_pts, train_pts):
			A = [] 

			for qpt, tpt in zip(query_pts, train_pts):
				A.append([0, 0, 0, -qpt[0], -qpt[1], -1, tpt[1]*qpt[0], tpt[1]*qpt[1], tpt[1]])
				A.append([qpt[0], qpt[1], 1, 0, 0, 0, -tpt[0]*qpt[0], -tpt[0]*qpt[1], -tpt[0]])
			
			_ , sigma , vtranspose = np.linalg.svd(A) # check if A need to be converted to numpy
			# print(f'sigma {sigma}') # sigma is not a diagonal matrix, it is returned as a 1D array
			# sigma1, sigma2 = sigma[0], sigma[-1]
			# ratio = sigma1/sigma2
			# print(f'ratio {ratio}')
			H = vtranspose[-1, :] # 9 element vector
			H = H.reshape(3,3) # 3x3 matrix
			H = H / H[-1][-1]
			return H

	def getSIFTFeatures(self, im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.sift.detectAndCompute(gray, None)
		return {'kp':kp, 'des':des}