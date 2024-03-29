"""
Official Python implementation of "A fast horizon detector and a new annotated dataset for maritime video processing",
a paper currently under review by Journal of Image and Graphics (United Kingdom).

Implementation by Yassir Zardoua
"""
import cv2
import cv2 as cv
import numpy as np
import os
from time import time
from warnings import warn
from math import pi, atan, sin, cos
import tkinter as tk


class FastHorizon:
    def __init__(self, init_all=True, canny_th1=25, canny_th2=45, Th_ROI=2, Th_slope=0.57, N_c=15, N_d=200,
                 D_Y_hl_th=50, D_alpha_hl_th=2, max_outliers_th=4, hough_D_rho=2, hough_D_theta=pi * (1 / 180),
                 resize_factor=0.6):
        """
        :param init_all: True if use want to reset all attributes. This argument is useful because I do not want to
        initialize (reset) some attributes, which I put in the if statement 'if init_all'.
        :param canny_th1: lower threshold of Canny's edge detector
        :param canny_th2: higher threshold of Canny's edge detector
        :param Th_ROI: parameter controlling the width of the ROI filter. Increasing Th_ROI increases mentioned width
        :param Th_slope: segments whose slope is higher than Th_slope are filtered out.
        :param N_c: number of the longest segments that are directly considered as candidate segments.
        :param N_d: number of the longest segments to evaluate against ROIs encompassing the longest N_c segments. The
        width of such ROIs is defined by parameter Th_ROI.
        :param D_Y_hl_th: the difference in position Y of current detected horizon and previous horizon is computed. If
        this difference exceeds the value of this parameter, the horizon is considered as an outlier.
        :param D_alpha_hl_th: as with DY_th, the difference in angle alpha of current detected horizon and previous
        horizon is computed. If this difference exceeds the value of this parameter, the horizon is considered as an
        outlier.
        :param max_outliers_th: the maximum number of consecutive outliers that if exceeded, it would indicate that the
        algorithm is locked into detecting consecutive wrong outliers.
        :param hough_D_rho: rho resolution of the hough transform. It is set to be equal to Th_ROI.
        :param hough_D_theta: theta resolution of the hough transform.
        :param resize_factor: the factor by which the width and height of processed image is multiplied; downsizing
        processed images requires resize factor to be in the interval ]0, 1[.
        """
        if init_all:
            self.resize_factor = resize_factor
            self.canny_th1 = canny_th1
            self.canny_th2 = canny_th2

            # Hyer-parameters
            self.Th_ROI = Th_ROI * self.resize_factor
            self.Th_slope = Th_slope
            self.N_c = N_c
            self.N_d = N_d
            self.N_d_org = N_d  # self.N_d may change due to the command 'self.N_d = self.Segs_d.shape[0]'. Thus, we
            # use self.N_d_org to update self.N_d to its original value for each new frame.

            # Outlier handler
            self.DY_th = D_Y_hl_th  # threshold to which to compare self.DY.
            self.Dphi_th = D_alpha_hl_th  # threshold to which to compare self.DY.
            self.Nth_F_out = max_outliers_th  # the maximum number of consecutive outliers that if exceeded,
            # it would indicate that the algorithm is locked into detecting consecutive wrong outliers.

            # Hough transform
            self.hough_D_rho = Th_ROI
            self.hough_D_theta = hough_D_theta

            # Instance of a Fast Segment Detector
            self.fsd = cv.ximgproc.createFastLineDetector(_canny_th1=self.canny_th1, _canny_th2=self.canny_th2)
            self.img_edges = np.zeros(shape=(100, 100), dtype=np.uint8)

            # Attributes related to outlier detections
            self.Y_prv = np.nan  # position of the horizon on the previous frame
            self.phi_prv = np.nan  # tilt of the horizon on the previous frame
            self.DY = np.nan  # the absolute difference between the current and previous position of the horizon
            self.Dphi = np.nan  # the absolute difference between the current and previous tilt of the horizon
            self.N_F_out = 0  # number of consecutive outliers that the outlier detector detected
            # constants
            self.color_red = (0, 0, 255)
            self.color_blue = (255, 0, 0)
            self.color_green = (0, 255, 0)
            self.color_yellow = (0, 255, 255)
            self.color_aqua = (255, 255, 0)
            self.D_rho = 1
            self.D_theta = 1 * (pi / 180)

        self.xe_f_n = None
        self.Len_f_n = None
        self.B_c = None
        self.DYe = None
        self.DYs = None

        self.Len_a = None
        self.Len_b = None
        self.Len_c = None
        self.Len_d = None
        self.Len_e = None
        self.Len_f = None
        self.Len_b_sort_idxs = None
        self.Len_f_n = None

        self.N_a = None
        self.N_b = None
        self.N_e = None
        self.N_f = None

        self.Q = None
        self.Qs = None
        self.Qe = None
        self.q = None

        self.Segs_a = None
        self.Segs_b = None
        self.Segs_c = None
        self.Segs_d = None
        self.Segs_e = None
        self.Segs_f = None

        self.Ye_d = None
        self.Ys_d = None

        # slopes
        self.alpha_a = None
        self.alpha_c = None

        # indexes
        self.b_from_a_idxs = None
        self.c_from_b_idxs = None
        self.d_from_b_idxs = None
        self.e_from_d_idxs = None

        # flags
        self.F_continue = True  # Subsequent processes are executed only if this flag is True
        self.F_det = True  # False means no horizon is deected
        self.F_out = False  # a flag whose truth indicates that detected horizon is an outlier

        self.hough_lines = None

        # images
        self.img_width = None
        self.img_edges = None
        self.in_img_bgr = None
        self.in_img_red = None  # the red channel after down-sizing (if resizing factor < 1)
        self.img_with_hl = None
        self.img_with_segs = None  # overlayed with different segment sets.

        # outputs
        self.Y = np.nan  # detected position of the horizon in pixels
        self.phi = np.nan  # detected tilt of the horizon in radians α
        self.theta = np.nan
        self.rho = np.nan
        self.latency = np.nan
        # self.det_horizons_per_file = None  # an array of shape (n, 4); where n is number_of_video_frames ( or the
        # number of rows in a gt .npy file), 4 columns for detected position (Y_det), detected angle (alpha_det),
        # |Y_GT-Y_det|, |alpha_GT - alpha_det|.

        self.u = None
        self.u_n = None
        self.x_n = None
        self.x_out = None
        self.y_out = None

        self.xe_a = None
        self.xe_b = None
        self.xe_c = None
        self.xe_d = None
        self.xe_f = None
        self.xe_f_n = None
        self.xe_hl = None

        self.xs_a = None
        self.xs_b = None
        self.xs_c = None
        self.xs_d = None
        self.xs_f = None
        self.xs_f_n = None
        self.xs_hl = None

        self.y_n = None
        self.y_out = None

        self.ye_a = None
        self.ye_b = None
        self.ye_c = None
        self.ye_d = None
        self.ye_f = None
        self.ye_f_n = None
        self.ye_hl = None

        self.ys_a = None
        self.ys_b = None
        self.ys_c = None
        self.ys_d = None
        self.ys_f = None
        self.ys_f_n = None
        self.ys_hl = None

        self.gt_position_hl = None  # ground truth position (position)
        self.gt_tilt_hl = None  # ground truth tilt (alpha)

    def get_horizon(self, img, get_image=False):
        """
        :param img:
        :param get_image: if True, the horizon is drawn on the attribute 'self.img_with_hl'
        :return:
        """
        self.start_time = time()
        self.__init__(init_all=False)
        self.N_d = self.N_d_org
        self.F_det = True  # will be set to false if no edge point is detected, which means that no horizon is detected
        self.get_horizon_edges(img=img)
        self.hough_transform()
        self.outlier_handler_module()
        self.linear_least_square_fitting()
        # self.outlier_hl_handler()  # check if the detection is outlier

        if self.F_det:  # we check this flag again because it can be changed in self.outlier_handler_module()
            print("Y = {}, phi = {}".format(self.Y, self.phi))
            self.Y_prv, self.phi_prv = self.Y, self.phi
            self.end_time = time()
            self.latency = round((self.end_time - self.start_time), 4)
        else:
            self.img_edges = np.zeros(shape=self.in_img_red.shape,
                                      dtype=np.uint8)  # create an edge map with no edge point
            self.Y = np.nan
            self.phi = np.nan
            self.latency = np.nan
            self.img_with_hl = img
        print("time execution is: ", self.latency, " seconds")
        return self.Y, self.phi, self.latency, self.F_det

    def get_horizon_edges(self, img):
        """
        Get xy coordinates of horizon edge pixels.
        :param img: input image
        :return: tuple (x_out, y_out)
        """
        self.x_out, self.y_out = None, None  # Initializing the output to None

        self.in_img_bgr = img
        self.img_with_hl = self.in_img_bgr.copy()

        # establishing the image to process: the resized red channel
        if self.resize_factor < 1:
            self.in_img_red = cv.resize(self.in_img_bgr[:, :, 2], dsize=(self.res_width, self.res_height))
        else:
            self.in_img_red = self.in_img_bgr[:, :, 2]

        self.F_continue = True  # when this flag is false, the next step of the horizon edge filter is not performed
        # and the last intermediate output (e.g., self.Segs_b) is assigned to output segments of the ROIF 'self.Segs_f'

        if self.N_c > self.N_d:
            raise ValueError("The input parameter N_c must be smaller than N_d")
        self.Segs_a = self.fsd.detect(self.in_img_red)

        if self.Segs_a is None:  # True if no segment is detected
            print("No segment is detected")
            self.x_out, self.y_out = None, None
            self.F_det = False
            return self.x_out, self.y_out
        self.lsf()
        if self.F_continue:
            self.roif()
        self.step()
        return self.x_out, self.y_out

    def linear_least_square_fitting(self):
        if self.F_det:
            self.get_inlier_edges()
            self.inlier_edges_xy = np.zeros((self.inlier_edges_x.size, 2), dtype=np.int32)
            self.inlier_edges_xy[:, 0], self.inlier_edges_xy[:, 1] = self.inlier_edges_x, self.inlier_edges_y
            [vx, vy, x, y] = cv.fitLine(points=self.inlier_edges_xy, distType=cv.DIST_L2,
                                        param=0, reps=1, aeps=0.01)
            self.hl_slope = float(vy / vx)  # float to convert from (1,) float numpy array to python float
            self.hl_intercept = float(y - self.hl_slope * x)

            self.xs_hl = int(0)
            self.xe_hl = int(self.org_width - 1)
            self.ys_hl = int(self.hl_intercept)  # = int((self.hl_slope * self.xs_hl) + self.hl_intercept)
            self.ye_hl = int((self.xe_hl * self.hl_slope) + self.hl_intercept)

            self.phi = (-atan(self.hl_slope)) * (180 / pi)  # - because the y axis of images goes down
            self.Y = ((((self.org_width - 1) / 2) * self.hl_slope + self.hl_intercept))

    def get_inlier_edges(self):
        """
        Process is described in inlier_edges.pdf file attached with this code project.
        """
        self.y_j, self.x_j = np.where(self.img_edges == 255)
        theta_p = self.theta + self.D_theta
        theta_n = self.theta - self.D_theta
        self.x_cte = 0.5 * (np.cos(theta_p) - np.cos(theta_n))
        self.y_cte = 0.5 * (np.sin(theta_p) - np.sin(theta_n))

        self.D_rho_j = np.abs(np.add(np.multiply(self.x_j, self.x_cte), np.multiply(self.y_j, self.y_cte)))
        self.D_rho_g = np.add(self.D_rho_j, self.D_rho)

        self.rho_j = np.add(np.multiply(self.x_j, np.cos(self.theta)), np.multiply(self.y_j, np.sin(self.theta)))
        inlier_condition = np.logical_and(self.rho_j <= (self.rho + self.D_rho_g / 2),
                                          self.rho_j >= (self.rho - self.D_rho_g / 2))

        self.inlier_edges_indexes = np.where(inlier_condition)
        self.inlier_edges_x = self.x_j[self.inlier_edges_indexes]
        self.inlier_edges_y = self.y_j[self.inlier_edges_indexes]
        self.inlier_edges_map = np.zeros(shape=self.img_edges.shape, dtype=np.uint8)
        self.inlier_edges_map[self.inlier_edges_y, self.inlier_edges_x] = 255

    def lsf(self):
        """
        Implements the Length-Slope Filter (LSF) stage.
            Required input(s): self.Segs_b
            Required parameter(s): self.N_c, self.N_d
            Computes: self.Segs_c, self.Segs_d
        """
        self.N_a = self.Segs_a.shape[0]  # get the number of all detected segments: self.N_a
        self.Segs_a = np.reshape(self.Segs_a, newshape=(self.N_a, 4))  # reshape from (N_a, 1, 4) to (N_a, 4). This
        # is easier for consequent processing.

        # Start of Slope Filter Portion
        self.xs_a, self.ys_a = self.Segs_a[:, 0], self.Segs_a[:, 1]
        self.xe_a, self.ye_a = self.Segs_a[:, 2], self.Segs_a[:, 3]
        self.alpha_a = np.divide(np.subtract(self.ye_a, self.ys_a), np.subtract(self.xe_a, self.xs_a))  # compute slopes
        # of segments in self.Segs_a
        self.b_from_a_idxs, = np.where(np.abs(self.alpha_a) < 0.58)  # We care about the first element of the returned
        # tuple
        self.Segs_b = self.Segs_a[self.b_from_a_idxs]
        self.N_b = self.Segs_b.shape[0]
        # End Slope Filter Portion

        if self.N_b <= self.N_c:
            self.Segs_f = self.Segs_b  # if the number of segments output by the slope filter is smaller than the
            # DESIRED number of segments in Segs_c, there is no point in performing the ROIF. Output segments
            # self.Segs_f will directly be equal to self.Segs_b.
            self.F_continue = False
            return

        # Start of Length Filter Portion
        self.xs_b, self.ys_b = self.Segs_b[:, 0], self.Segs_b[:, 1]
        self.xe_b, self.ye_b = self.Segs_b[:, 2], self.Segs_b[:, 3]
        self.Len_b = np.sqrt(np.add(np.square(np.subtract(self.xs_b, self.xe_b)),
                                    np.square(np.subtract(self.ys_b, self.ye_b))))

        self.Len_b_sort_idxs = np.flip(np.argsort(self.Len_b))
        self.c_from_b_idxs = self.Len_b_sort_idxs[0:self.N_c]
        self.Segs_c = self.Segs_b[self.c_from_b_idxs]
        self.d_from_b_idxs = self.Len_b_sort_idxs[self.N_c:self.N_c + self.N_d]
        self.Segs_d = self.Segs_b[
            self.d_from_b_idxs]  # Note: if self.N_d > self.N_b, this command becomes equivalent to: self.
        self.N_d = self.Segs_d.shape[0]  # update of self.N_d is performed because self.N_d may be have been > self.N_b
        # End of Length Filter Portion
        return

    def roif(self):
        self.xs_c, self.ys_c = self.Segs_c[:, 0], self.Segs_c[:, 1]
        self.xe_c, self.ye_c = self.Segs_c[:, 2], self.Segs_c[:, 3]
        self.alpha_c = self.alpha_a[self.b_from_a_idxs][self.c_from_b_idxs]
        self.B_c = np.subtract(self.ys_c, np.multiply(self.alpha_c, self.xs_c))
        self.B_c = np.broadcast_to(np.reshape(self.B_c, newshape=(self.N_c, 1)), shape=(self.N_c, self.N_d))

        self.xs_d, self.ys_d = self.Segs_d[:, 0], self.Segs_d[:, 1]
        self.xe_d, self.ye_d = self.Segs_d[:, 2], self.Segs_d[:, 3]

        self.Ys_d = np.broadcast_to(np.reshape(self.ys_d, newshape=(1, self.N_d)), shape=(self.N_c, self.N_d))
        self.Ye_d = np.broadcast_to(np.reshape(self.ys_d, newshape=(1, self.N_d)), shape=(self.N_c, self.N_d))

        self.alpha_c = np.reshape(self.alpha_c, newshape=(self.N_c, 1))  # reshape is necessary for homogeneity of
        # multiplied vectors

        self.DYs = np.abs(np.subtract(np.add(np.multiply(self.alpha_c, self.xs_d), self.B_c), self.Ys_d))
        self.DYe = np.abs(np.subtract(np.add(np.multiply(self.alpha_c, self.xe_d), self.B_c), self.Ye_d))

        self.Qs = np.less_equal(self.DYs, self.Th_ROI)
        self.Qe = np.less_equal(self.DYe, self.Th_ROI)
        self.Q = np.logical_and(self.Qs, self.Qe)
        self.q = np.any(self.Q, axis=0)  # performs logical or along the rows axis <==> operation applied on elements
        # of the same columns
        self.e_from_d_idxs, = np.where(self.q == True)
        self.N_e = self.e_from_d_idxs.shape[0]
        if self.N_e > 0:  # True if at least one segment survived the ROIF.
            self.Segs_e = self.Segs_d[self.e_from_d_idxs]
        else:
            self.Segs_e = np.zeros((0, 4))
        self.Segs_f = np.concatenate((self.Segs_c, self.Segs_e), axis=0)
        return

    def step(self):
        self.x_out = np.zeros((0,))  # initialize coordinates of output edge pixels
        self.y_out = np.zeros((0,))  # initialize coordinates of output edge pixels
        self.N_f = self.Segs_f.shape[0]
        if self.N_f == 0:  # True if no segment survived the filter.
            self.F_det = False
            return
        self.xs_f, self.ys_f = self.Segs_f[:, 0], self.Segs_f[:, 1]
        self.xe_f, self.ye_f = self.Segs_f[:, 2], self.Segs_f[:, 3]

        # computing self.Len_f
        if self.F_continue:  # True
            # self.Segs_f != self.Segs_b. Thus, get self.Len_f from self.Len_b through indexing
            self.Len_c = self.Len_b[self.c_from_b_idxs]
            self.Len_e = self.Len_b[self.d_from_b_idxs][
                self.e_from_d_idxs]  # self.Len_b[self.d_from_b_idxs] = self.Len_d
            self.Len_f = np.concatenate((self.Len_c, self.Len_e))
        else:
            # else: self.Segs_f = self.Segs_b. Thus, self.Segs_f can't be obtained through indexing because self.Segs_c
            # and self.Segs_e are empty.
            self.Len_f = np.sqrt(np.add(np.square(np.subtract(self.xs_f, self.xe_f)),
                                        np.square(np.subtract(self.ys_f, self.ye_f))))

        self.Len_f = np.uint16(np.subtract(self.Len_f, 1))  # All length values must be integers to allow subsequent pr-
        # ocessing. Subtracting 1 allows subtraction of 1 for N_f iteration. See Eq1 in handwritten theory: STP, page 2
        # for a recall.
        self.u = np.arange(0, self.Len_f[0])  # the first element of self.Len_f is the length of the longest segment
        for self.Len_f_n, self.xs_f_n, self.ys_f_n, self.xe_f_n, self.ye_f_n in \
                zip(self.Len_f, self.xs_f, self.ys_f, self.xe_f, self.ye_f):
            self.u_n = self.u[0:self.Len_f_n]
            self.x_n = np.add(np.multiply(np.divide(np.subtract(self.xe_f_n, self.xs_f_n), self.Len_f_n), self.u_n),
                              self.
                              xs_f_n)
            self.y_n = np.add(np.multiply(np.divide(np.subtract(self.ye_f_n, self.ys_f_n), self.Len_f_n), self.u_n),
                              self.
                              ys_f_n)

            # updating output coordinates of candidate edge pixels.
            self.x_out = np.uint16(np.concatenate((self.x_out, self.x_n)))
            self.y_out = np.uint16(np.concatenate((self.y_out, self.y_n)))

    def draw_segs(self, img_dst, segments, colors, thickness=4):
        """
        Description:
        ------------
        Draws line segments given in 'segments' on 'img_dst'.

        Parameters:
        -----------
        :param img_dst: image on which to draw segments
        :param segments: a list of numpy arrays of shape (N,4). Each row in such array contains: xs, ys, xe, ye, where:
            x and y: refer to x and y coordinates, respectively.
            s and e: refer to starting and endpoints, respectively.
        :param colors: a list containing color values that correspond to each element in the tuple 'segments'. Each el-
        ement in list 'colors' is a tuple (Blue, Green, Red) or a scalar.
        :param thickness: thickness of segments to draw

        Usage Example:
        --------------
            segments = [self.Segs_a, self.Segs_b]
            colors = [(0, 0, 255), (0, 255, 0)]
            self.img_segs = self.draw_segs(img_dst=self.in_img_bgr, segments=segments, colors=colors)
            cv.imwrite("result.png", self.img_segs)

        :return: img_dst with drawn segments
        """
        img_dst = np.float32(img_dst)
        for seg, color in zip(segments, colors):
            seg = np.int32(seg)
            for points in seg:
                cv.line(img_dst, (points[0], points[1]), (points[2], points[3]), color=color,
                        thickness=thickness)
        return np.uint8(img_dst)

    def draw_points(self, img_dst, points_x, points_y, radius=0, color=(0, 0, 255)):
        """
        img, center, radius, color, thickness=None
        :param img_dst:
        :param points_x
        :param points_y
        :param radius: radius of points to draw
        :param color: a tuple of three elements
        :return:
        """
        try:
            for point_x, point_y in zip(points_x, points_y):
                cv.circle(img_dst, (point_x, point_y), radius=radius, color=color, thickness=-1)
        except:  # except any error
            warn("Drawing of points either could not be started or not completed.")
        return img_dst

    def hough_transform(self):
        if not self.F_det:  # True if the edge map contains edges. False otherwise (zero edges detected)
            return
        self.img_edges = np.zeros(shape=self.in_img_red.shape, dtype=np.uint8)
        self.img_edges[self.y_out, self.x_out] = 255
        if self.resize_factor < 1:
            self.img_edges = cv.resize(self.img_edges, dsize=(self.org_width, self.org_height))
            # self.img_edges = cv.Canny(self.img_edges, 254, 254)
            self.img_edges = np.int16(self.img_edges)
            # self.img_edges = cv.threshold(self.img_edges, thresh=254, type=cv.THRESH_BINARY, maxval=255)[1]
            self.img_edges = cv.Canny(self.img_edges, self.img_edges, 254, 254)
            # cv.imwrite("thresholded edges.png", self.img_edges)
            # exit()
            # self.img_edges = cv.ximgproc.thinning(self.img_edges)

        self.hough_lines = cv.HoughLines(image=self.img_edges, rho=self.hough_D_rho, theta=self.hough_D_theta,
                                         threshold=2, min_theta=np.pi / 3, max_theta=np.pi * (2 / 3))
        if self.hough_lines is not None:  # executes if Hough detects a line
            self.F_det = True
        else:  # True if no line is detected from the Hough transform
            self.phi = np.nan
            self.Y = np.nan
            self.latency = np.nan
            self.F_det = False

    def outlier_handler_module(self):
        if not self.F_det:
            return  # execute this method only if there is at least one line in the Hough space.
        self.outlier_checker()
        self.outlier_replacer()
        self.failure_state_handler()

    def outlier_checker(self):
        """
        Checks if the longest Hough peak, which we assume as the rough horizon, is an outlier. In such case, flag
        self.F_out is set to True.
        """
        self.F_out = False  # reset to false
        # compute (Y, phi) corresponding to the longest Hough line. # # # # # # # # # # # # # # # # # # # # # # # #
        self.rho, self.theta = self.hough_lines[0][0]  # self.theta in radians
        self.phi = ((np.pi / 2) - self.theta)
        self.phi = self.phi * (180 / np.pi)  # conversion to degrees
        self.img_width = self.img_edges.shape[1]
        self.Y = (self.rho - 0.5 * self.img_width * np.cos(self.theta)) / (np.sin(self.theta))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # check if computed parameters (Y, phi) are outliers # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # computing difference parameters
        self.DY = abs(self.Y - self.Y_prv)
        self.Dphi = abs(self.phi - self.phi_prv)
        # establishing outlier flag
        self.F_out = (self.DY > self.DY_th) or (self.Dphi > self.Dphi_th)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def outlier_replacer(self, M=2):
        """
        Finds a valid substitute to the outlier horizon.
        :param M: the number of substitute candidates to consider for replacing the outlier horizon. If M = -1, all
        Hough candidates will be considered. The substitute is the line with the smallest position difference, self.DY,
        that correspond to self.Dphi < self.Dphi_th.
        """
        if not self.F_out:
            return  # execute only if the detection is outlier
        self.img_width = self.img_edges.shape[1]
        self.hough_lines_nbr = self.hough_lines.shape[0]  # nbr of candidate lines in the hough space
        self.hough_lines = np.reshape(self.hough_lines, newshape=(self.hough_lines_nbr, 2))
        # get polar parameters of the next M candidates
        if M == -1:
            self.rho_cands = self.hough_lines[1::][:, 0]
            self.theta_cands = self.hough_lines[1::][:, 1]
        else:
            self.rho_cands = self.hough_lines[1: (M + 1)][:, 0]
            self.theta_cands = self.hough_lines[1: (M + 1)][:, 1]
        # convert (rho, theta)  to (Y, phi)
        self.phi_cands = np.subtract((np.pi / 2), self.theta_cands) * (180 / np.pi)  # * (180 / np.pi) converts to degs
        self.Y_cands = (self.rho_cands - 0.5 * self.img_width * np.cos(self.theta_cands)) / (np.sin(self.theta_cands))
        # computing parameters of selected candidate lines.
        self.DY_cands = np.abs((np.subtract(self.Y_cands, self.Y_prv)))
        self.Dphi_cands = np.abs((np.subtract(self.phi_cands, self.phi_prv)))

        self.Isrt_cands = np.argsort(self.DY_cands)  # get indices that would sort self.DY_cands from lowest to highest

        self.DYsrt_cands = self.DY_cands[self.Isrt_cands]  # self.DYsrt_cands is the sorted version of self.DY_cands
        self.Dphisrt_cands = self.Dphi_cands[self.Isrt_cands]  # array self.Dphi_cands corresponding to self.DYsrt_cands
        self.Ysrt_cands = self.Y_cands[self.Isrt_cands]  # positions Y corresponding to self.DYsrt_cands
        self.phisrt_cands = self.phi_cands[self.Isrt_cands]  # angles phi corresponding to self.DYsrt_cands

        self.ISub = np.logical_and(self.DYsrt_cands < self.DY_th, self.Dphisrt_cands < self.Dphi_th)
        self.Isub = np.where(self.ISub == True)[0]  # contains indices corresponding to True values in self.ISub
        self.sub_nbr = self.Isub.shape[0]  # number of valid substitutes
        if self.sub_nbr > 0:
            # print("A replacement has been performed")
            self.Isub = self.Isub[0]  # the substitute index is the first value of vector self.Isub, which corresponds
            # to the Hough line with the smallest position and tilt difference
            self.Y = self.Ysrt_cands[self.Isub]
            self.phi = self.phisrt_cands[self.Isub]

            # finding corresponding polar parameters
            self.rhosrt_cands = self.rho_cands[self.Isrt_cands]
            self.thetasrt_cands = self.theta_cands[self.Isrt_cands]

            self.rho = self.rhosrt_cands[self.Isub]
            self.theta = self.thetasrt_cands[self.Isub]
            self.theta = (90 - self.phi) * (pi / 180)  # in rads, required to compute self.rho
            self.rho = (self.Y * sin(self.theta)) + (0.5 * self.img_width * cos(self.theta))
            # print("corresponding polar pars: rho = {}; theta = {}".format(self.rho, self.theta))
            # print("corresponding polar pars indexed: rho = {}; theta = {}".format(self.rho_indexed, self.theta_indexed))
            # self.theta = self.theta * (180/pi)
        else:  # True if there is no substitute
            self.F_det = False  # no substitute means no horizon is detected <==> set flag to False.

    def failure_state_handler(self):
        """
        Handle the failure state; being locked into detecting consecutive outliers. Solves this issue by setting
        previous detections to np.nan will move the algorithm out of this situation because self.D self.DY and self.Dphi
        will be equal to nan. This will make condition self.F_out False no matter the value of position and angle
        thresholds (self.DY_th, self.Dphi_th). In this way, we will avoid finding a substitute line to the Hough peak,
        which we assume to be the correct horizon.
        """
        if self.F_out:
            self.N_F_out += 1
            if self.N_F_out > self.Nth_F_out:  # True indicates that the algorithm has been locked into the situation of
                # detecting self.Nth_F_out consecutive outliers Setting previous detections to np.nan will move the
                # algorithm out of this situation because self.D self.DY and  self.Dphi will be equal to nan. This will
                # make condition self.F_out False no matter the value of position and angle thresholds
                # (self.DY_th, self.Dphi_th)
                self.Y_prv = np.nan
                self.phi_prv = np.nan
                self.N_F_out = 0
        else:
            self.N_F_out = 0

    def draw_hl(self):
        """
        Draws the horizon line on attribute 'self.img_with_hl'
        """
        if self.F_det:
            # thickness = int(5 * self.resize_factor)  # make thickness invariant to resolution change
            thickness = 5
            cv.line(self.img_with_hl, (self.xs_hl, self.ys_hl), (self.xe_hl, self.ye_hl), (0, 0, 255),
                    thickness=thickness)

    def y_from_xpolar(self, x, rho, theta):
        """
        returns y coordinate on the line defined by polar coordinates rho and theta, and that corresponds to x
        """
        return int((1 / np.sin(theta)) * (rho - x * np.cos(theta)))

    def reset_for_new_video(self):
        """
        resets attributes related to the previous video so we can avoid processing the new video using results from
        another video.
        """
        # attributes related to outlier detections
        self.Y_prv = np.nan  # position of the horizon on the previous frame
        self.phi_prv = np.nan  # tilt of the horizon on the previous frame
        self.DY = np.nan  # the absolute difference between the current and previous position of the horizon
        self.Dphi = np.nan  # the absolute difference between the current and previous tilt of the horizon
        self.N_F_out = 0  # number of consecutive outliers that the outlier detector detected

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                 draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds
        between 0 and 1) specifying the ratio of the diameter of the resized image being processed. For instance, if
        the attributre self.dsize = (640, 480), the threshold that will be used in the hough transform is sqrt(640^2 +
        480^2) * hough_threshold_ratio, rounded to the nearest integer.
        :param src_gt_folder: absolute path to the ground truth horizons corresponding to source video files.
        :param src_video_folder: absolute path to folder containing source video files to process
        :param dst_video_folder: absolute path where video files with drawn horizon will be saved.
        :param dst_quantitative_results_folder: destination folder where quantitative results will be saved.
        :param draw_and_save: if True, all detected horizons will be drawn on their corresponding frames and saved as video files
        in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            # yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
            #            " to proceed, otherwise, click on 'n'.\n"
            #            "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
            #            "files with similar names.")
            yn = 'y'
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        nbr_of_vids = len(src_video_names)
        vid_indx = 0
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            self.reset_for_new_video()
            vid_indx += 1
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
            self.res_width = int(self.org_width * self.resize_factor)
            self.res_height = int(self.org_height * self.resize_factor)
            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "Author1_" + src_video_name)
                if draw_and_save:
                    video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.org_width, self.org_height),
                                                  True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            #
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
            if nbr_of_frames != nbr_of_annotations:
                warning_text_1 = "The number of annotations (={}) does not equal to the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                print("----------WARNING---------")
                print(warning_text_1)
                print("--------------------------")

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            self.__init__(init_all=True)  # reinitialize all attribures of the algorithm before processing the new video
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.input_img = frame
                self.get_horizon(img=self.input_img)  # gets the horizon position and
                # tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon[0], gt_horizon[1]
                # print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}".
                #       format(self.Y, self.gt_position_hl, self.phi, self.gt_tilt_hl))
                print("Frame {}/{}. Video {}/{}".format(idx, nbr_of_frames, vid_indx, nbr_of_vids))
                self.det_horizons_per_file[idx] = [self.Y,
                                                   self.phi,
                                                   round(abs(self.Y - self.gt_position_hl), 4),
                                                   round(abs(self.phi - self.gt_tilt_hl), 4),
                                                   self.latency]
                if draw_and_save:
                    self.draw_hl()  # draws the horizon on self.img_with_hl
                    video_writer.write(self.img_with_hl)
            cap.release()
            if draw_and_save:
                video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))

            # saving the .npy file of quantitative results of current video file # # # # # # # # # # # # # # # # # # # #
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                          src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                         "all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)

        self.Y_hl_all = self.det_horizons_all_files[:, 2]
        self.alpha_hl_all = self.det_horizons_all_files[:, 3]
        self.latency_all = self.det_horizons_all_files[:, 4]
        self.false_positive_nbr = np.size(np.argwhere(np.isnan(self.Y_hl_all)))

    def video_demo(self, video_path, display=True):
        demo_cap = cv2.VideoCapture(video_path)
        self.org_width = int(demo_cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
        self.org_height = int(demo_cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
        self.res_width = int(self.org_width * self.resize_factor)
        self.res_height = int(self.org_height * self.resize_factor)

        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Get the screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        if (self.org_height >= screen_height) or (self.org_width >= screen_width):
            dst_width = int(screen_width - 0.2 * screen_width)
            dst_height = int(screen_height - 0.2 * screen_height)
        else:
            dst_width = self.org_width
            dst_height = self.org_height

        wait = 30
        if not demo_cap.isOpened():
            print("Error: Could not open video file.")
            return

        try:
            while True:
                ret, frame = demo_cap.read()

                if not ret:
                    break
                self.input_img = frame
                self.get_horizon(img=self.input_img)  # gets the horizon position and
                # tilt
                self.draw_hl()  # draws the horizon on self.img_with_hl
                if display:
                    # video_writer.write(self.img_with_hl)

                    cv2.imshow("Horizon Detection", cv2.resize(self.img_with_hl, (dst_width, dst_height)))
                    if cv2.waitKey(wait) & 0xFF == ord('q'):
                        break

        finally:
            # Release the video capture object
            demo_cap.release()
            root.destroy()
            # Close all OpenCV windows
            if display:
                cv2.destroyAllWindows()
