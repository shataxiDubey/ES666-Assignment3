import numpy as np
import cv2
from src.ShataxiDubey.matchers import matchers
import time
import os
import glob


# For reference, I have used the code from the source https://kushalvyas.github.io/stitching.html
# I have applied homography to right image to get coordinates with respect to left image.

class PanaromaStitcher:
	def __init__(self):
		self.images = []
		self.count = 0
		self.left_list, self.right_list, self.center_im = [], [],None
		self.homography_matrix_list = []
		self.matcher_obj = matchers()

	def prepare_lists(self):
		self.centerIdx = self.count/2 
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<=self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])

	def leftshift(self):
		# here we are shifting everything by offset because we want the coordinates to lie in positive coordinate system
		# also offset is also added in the final canvas size so that stitched images can be accomodated
		# self.left_list = reversed(self.left_list)
		a = self.left_list[0]
		for b in self.left_list[1:]:
			H = self.matcher_obj.match(a, b, 'left')
			self.homography_matrix_list.append(H)
			print("Homography is : ", H)
			xh = np.linalg.inv(H)
			print("Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
			ds = ds/ds[-1]
			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			print("final ds after transaltion=>", ds, np.dot(xh, np.array([0,0,1])))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
			print("canvas dsize =>", dsize)
			tmp = cv2.warpPerspective(a, xh, dsize)
			tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			a = tmp

		self.leftImage = tmp

		
	def rightshift(self):
		for each in self.right_list:
			H = self.matcher_obj.match(self.leftImage, each, 'right')
			self.homography_matrix_list.append(H)
			print( "Homography :", H)
			txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
			tmp = cv2.warpPerspective(each, H, dsize)
			tmp = self.mix_and_match(self.leftImage, tmp)
			print("tmp shape",tmp.shape)
			self.leftImage = tmp
		

	def mix_and_match(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		for i in range(0, i1x):
			for j in range(0, i1y):
				# when both warped and reference image have black pixels then warp image will also have black pixel
				if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
					warpedImage[j,i] = [0, 0, 0]
				else: # when reference image have non black pixel then add non black pixel in the warped image
					warpedImage[j, i] = leftImage[j,i]
		return warpedImage
	
	
	def make_panaroma_for_images_in(self, path):
		imf = path
		all_images = sorted(glob.glob(imf+os.sep+'*'))
		print('Found {} Images for stitching'.format(len(all_images)))
		img_arr = []
		for img in all_images:
			img_arr.append(cv2.imread(img))
		self.images = img_arr
		self.count = len(self.images)
		self.prepare_lists()
		self.leftshift()
		self.rightshift()
		return self.leftImage, self.homography_matrix_list
        
# for I1 and I2, the stitched of three images resulted in very big size so stitched upto three images only.
# for I3,I4,I5,I6, image stitching is done for all images      
