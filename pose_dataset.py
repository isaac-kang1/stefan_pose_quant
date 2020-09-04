import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True, precision=6)
import os
import glob
import cv2
import sys
import time
import pickle
from random import shuffle
from pose_gt import *
from matplotlib import pyplot as plt
from time import time as t
from tqdm import tqdm


class Dataset(object):
    def __init__(self, args):
        self.args = args
        self.test_mode = (args.mode == 'test')
        self.cad_dir = args.cad_path
        self.asc_dir = args.point_cloud_path
        self.krt_dir = args.krt_path
        self.dataset_dir = args.dataset_path
        self.label_dir = os.path.join(args.dataset_path, 'output', 'label')
        self.test_dir = args.test_imgs_path
        self.view_dir = args.views_path
        self.background_dir = args.background_path
        self.load_filenames(self.krt_dir, self.view_dir, self.test_dir)
        self.classes_len = len(self.classes)
        self.scale = args.scale
        self.NUM_POINTS = args.ptcld_len
        self.batch_size = args.batch_size
        self.batch_index = 0

    def time_conversion(self, seconds):
        s = int(seconds % 60)
        minutes = seconds // 60
        m = int(minutes % 60)
        hours = minutes // 60
        h = int(hours % 24)
        d = hours // 24
        if d != 0:
            return '{}d {}h {}m {}s'.format(d, h, m, s)
        elif h != 0:
            return '{}h {}m {}s'.format(h, m, s)
        elif m != 0:
            return '{}m {}s'.format(m ,s)
        else:
            return '{}s'.format(s)

    ##### LABELS #####

    def load_filenames(self, krt_dir, view_dir, test_dir):
        """make list of filepaths
        
        k_adr : K
        rt_default_adr : default Rt
        rt_adrs : Rt
        classes : CAD filenames
        views_adrs : list of list of rendered images filenames
        test_adrs : list of list of test images filenames
        
        Args:
            krt_dir: {str} KRt path
            view_dir: {str} rendered images path
            test_dir: {str} test images path
        """
        # krt
        self.krt_adrs = sorted(glob.glob(krt_dir + '/*.txt'))
        self.k_adr = self.krt_adrs[0]
        self.rt_default_adr = self.krt_adrs[-1]
        self.rt_adrs = self.krt_adrs[1:-1]

        # cad classes
        self.classes = [os.path.basename(p) for p in sorted(glob.glob(self.dataset_dir + '/output/image/cropped_grouped/*'))]
        self.rendering_adrs = []
        for cad_name in self.classes:
            self.rendering_adrs.append(sorted(glob.glob(os.path.join(self.args.renderings_path, cad_name, '*'))))
        
        # read label txt
        with open(self.label_dir + '/pose_label.txt', 'r') as f:
            labels = [line for line in f.readlines()]

        # split train / val / test
        self.test_labels = labels[:100] # image number(10) x cad number(10)
        print('--------------------')
        print('number of test imgs :', len(self.test_labels))
        if not self.test_mode:
            train_labels = labels[100:] if self.args.train_size == 0 else labels[100:self.args.train_size]
            num_val = len(train_labels)//10 # 1/10 of train data -> val data
            self.val_labels = train_labels[:num_val]
            self.train_labels = train_labels[num_val:]
            print('number of validation images :', len(self.val_labels))
            print('number of train images :', len(self.train_labels))
        print('--------------------')

    def shuffle_and_parse_labels(self, labels, shuffle_labels=True):
        """shuffle labels, parse each label to img_lab, cad_lab, pose_lab        
        
        Args:
            labels: {list{str}} list of img_lab + cad_lab + pose_lab as string
            shuffle : {bool} wheter to shuffle
        
        Returns:
            img_labels : {list{str}} list of image paths
            cad_labels : {list{str}} list of cad names
            pose_labels : {list{int}} list of pose index
        """
        if shuffle_labels:
            shuffle(labels)
        cropped_img_labels = []
        cad_labels = []
        pose_labels = []
        for label in labels:
            parsed = label.split(',')
            cropped_img_labels.append(parsed[0])
            cad_labels.append(parsed[1])
            pose_labels.append(int(parsed[2].rstrip('\n')))
        return cropped_img_labels, cad_labels, pose_labels

    def manage_duplicate_pose(self, class_name_list, pose_idx_list):
        """Some cad files have same output rendering for different pose,
        due to symmetry of the object. For known cad objects that have
        symmetric properties, change pose label for pose with same rendering
        output with a smaller pose label, to the smaller pose label.
        
        Args:
            class_name_list: {list {str}} cad name list 
            pose_idx_list: {list {int}} pose index list
        
        Returns:
            
            modified_pose_idx_list : {list {int}} modified pose index list
        """
        modified_pose_idx_list = []
        for class_name, pose_idx in zip(class_name_list, pose_idx_list):
            duplicate = None
            if class_name == 'stefan_part1':
                duplicate = DUPLICATE_POSE_stefan_part1
            elif class_name == 'stefan_part2':
                duplicate = DUPLICATE_POSE_stefan_part2
            elif class_name == 'stefan_part3':
                duplicate = DUPLICATE_POSE_stefan_part3
            try:
                duplicate_list = list(duplicate[np.where(duplicate == pose_idx)[0][0]])  # ex) [0, 40]
            except:
                duplicate_list = [pose_idx]
            modified_pose_idx_list.append(duplicate_list[0])
        assert len(pose_idx_list) == len(modified_pose_idx_list), "AssertError : Length of modified/original pose index list doesn't match"
        return modified_pose_idx_list

    def batch_labels(self, img_lab, cad_lab, pose_lab, batch_size, batch_index):
        """return batched img_lab, cad_lab, pose_lab
                
        Args:
            img_lab: {list{str}} list of image paths
            cad_lab: {list{str}} list of cad names
            pose_lab: {list{int}} list of pose index
            batch_size : batch_size
            batch_index : batch_index
        
        Returns:
            batched versions of img_lab, cad_lab, pose_lab
        """
        b_img_lab = img_lab[batch_index * batch_size: (batch_index + 1) * batch_size]
        b_cad_lab = cad_lab[batch_index * batch_size: (batch_index + 1) * batch_size]
        b_pose_lab = pose_lab[batch_index * batch_size: (batch_index + 1) * batch_size]
        assert len(b_pose_lab) != 0, 'batch index out of range'
        return b_img_lab, b_cad_lab, b_pose_lab

    def return_gt_rt(self, b_pose_lab):
        b_rt_gt = []
        for pose_lab in b_pose_lab:
            rt_gt = np.expand_dims(np.loadtxt(self.rt_adrs[pose_lab]), 0)
            b_rt_gt = np.concatenate((b_rt_gt, rt_gt), 0) if len(b_rt_gt) else rt_gt
        return b_rt_gt

    ##### VIEW #####

    def resize_and_pad(self, img, a=150, random_border=False):
        """Make image similar to detection result
        
        Resize non-zero part of image to size (a, a).
        Then pad image to size (224, 224).
        Used for train batch (when adding data augmentation) and test batch

        Args:
            img: {uint8} image to be resized and padded 
            a: {int} resizing size before padding (default: {150})
            random_border : {bool} if True, add randomness to crop region
        
        Returns:
            {uint8} resized and padded image
        """
        # find object region
        non_zero = np.nonzero(255 - img)
        y_min = np.min(non_zero[0])
        y_max = np.max(non_zero[0])
        x_min = np.min(non_zero[1])
        x_max = np.max(non_zero[1])

        if random_border:
            h, w = img.shape[:2]
            # border distortion range
            dw = w // 10
            dh = h // 10
            # pad image
            img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, None, [255, 255, 255])
            # apply distortion (pad adjustment + distortion)
            x_min += dw + np.random.randint(-dw, dw)
            x_max += dw + np.random.randint(-dw, dw)
            y_min += dh + np.random.randint(-dh, dh)
            y_max += dh + np.random.randint(-dh, dh)

        img = img[y_min:y_max + 1, x_min:x_max + 1]
        # resize to 150, 150
        long_side = np.max(img.shape)
        ratio = a / long_side
        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation = cv2.INTER_AREA)
        # pad to 224, 224
        pad_left = int(np.ceil((224 - img.shape[1]) / 2))
        pad_right = int(np.floor((224 - img.shape[1]) / 2))
        pad_top = int(np.ceil((224 - img.shape[0]) / 2))
        pad_bottom = int(np.floor((224 - img.shape[0]) / 2))
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, [255, 255, 255])
        return img

    ##### CORNERPOINTS ######

    def read_cornerpoints(self, cad_lab):
        """acquire subset of size NUM_POINTS from obj file
        
        acquire points from obj data with constant stride
        scale down size
        
        Args:
            cad_lab: {str} filepath to cad file 
        
        Returns:
            cornerpoints {float : NUM_POINTS=50 x 3}
        """
        cad_filename = os.path.join(self.cad_dir, cad_lab + '.obj')
        with open(cad_filename ,'r') as f:
            cornerpoints = []
            while True:
                line = f.readline()
                if not line : break
                if not line.startswith('v') : continue
                line_split = line.split(' ')
                try:
                    line_split.remove('')
                except:
                    pass
                if len(line_split) == 4:
                    indicator, x, y, z = line_split
                    if indicator == 'v':
                        cornerpoints.append([x, y, z])
            cornerpoints = np.array(cornerpoints).astype(np.float32)
            stride = cornerpoints.shape[0] // self.NUM_POINTS
            index_list = list(range(0, cornerpoints.shape[0], stride))[:self.NUM_POINTS]
            cornerpoints = np.take(cornerpoints, index_list, axis=0)
            cornerpoints /= self.scale
        cornerpoints = np.expand_dims(cornerpoints, 0)

        return cornerpoints


    def transform_pointcloud(self, cornerpoints, RT):
        """apply RT transform to cornerpoints
        
        Args:
            cornerpoints: {float : self.NUM_POINTS=50 x 3} subset of pointcloud
            RT: RT calculated from blender
        
        Returns:
            transformed cornerpoints {float : self.NUM_POINTS=50 x 3}
        """
        ones = np.ones((cornerpoints.shape[0], 1))
        homogenous_coordinate = np.append(cornerpoints[:, :3], ones, axis=1) # Nx4 : (x, y, z, 1)
        # 3xN
        coord_3D = RT @ (homogenous_coordinate.T) # 3xN
        # Nx3 (x, y, z)
        coord_3D = coord_3D.T

        return coord_3D

    ##### POSE IMAGE #####

    def pointcloud_to_poseimg(self, ptcld, K, RT):
        """make image of pointcloud transformed with specific K, R, t
        
        Args:
            ptcld: {float: (None x 3)} point cloud data from .asc file or cornerpoints
            K: K calculated from blender
            RT: RT calculated from blender
        
        Returns:
            image of transformed pointcloud by (K, R, t)
        """

        ones = np.ones((ptcld.shape[0], 1))
        homogenous_coordinate = np.append(ptcld[:, :3], ones, axis=1) # Nx4 : (x, y, z, 1)
        homogenous_2D = K @ RT @ (homogenous_coordinate.T) # 3xN
        # 2xN
        coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
        # Nx2 (x, y)
        coord_2D = (coord_2D.T).astype(np.int)

        pose_img = np.zeros((224, 224), np.uint8)
        x = np.clip(coord_2D[:, 0], 0, 223)
        y = np.clip(coord_2D[:, 1], 0, 223)
        pose_img[y, x] = 255

        return pose_img


    def return_pose_comparison_img(self, img_lab_list, cad_lab_list, pose_lab_list, b_RT_gt, b_RT_pred, rt_closest_idx_list, pose_img_len):
        """make pose comparision image
        
        [input image, gt pose image, pred pose image, closest candidate pose image(pointcloud distance)
        ,closest candidate pose image(Rt distance), gt pose image (cornerpoints), pred pose image (cornerpoints)]
        
        Args:
            img_lab_list : {list {str}} path to cropped image file
            cad_lab_list : {list {str}} cad name
            pose_lab_list: {list {int}} pose index
            b_RT_gt: {float (None, 3, 4)} ground truth R, t
            b_RT_pred: {float (None, 3, 4)} predicted R, t
            rt_closest_idx_list: {list {int}} closest pose index (Rt distance)
            pose_img_len : {int} number of rows of output image
        
        Returns:
            {uint8 : (224 x pose_img_len, 224 x 7)} concatenated pose images
        """
        pose_img = []
        for img_lab, cad_lab, pose_lab, RT_gt, RT_pred, rt_closest_idx in zip(img_lab_list, cad_lab_list, pose_lab_list, b_RT_gt, b_RT_pred, rt_closest_idx_list):
            #---------------VIEW-----------------
            # view image
            view_filename = os.path.join(self.dataset_dir, img_lab)
            view = cv2.imread(view_filename, cv2.IMREAD_GRAYSCALE)
            view = self.resize_and_pad(view)
            cv2.putText(view, str(pose_lab).zfill(2), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            #----------------CORNER POINTS-----------------------
            cornerpoints = self.read_cornerpoints(cad_lab)[0]

            #----------------POINT CLOUD (asc)-----------------------
            asc_filename = os.path.join(self.asc_dir, cad_lab + '_SAMPLED_POINTS.asc')
            ptcld_asc = np.loadtxt(asc_filename, usecols=(0, 1, 2), skiprows=2)
            ptcld_asc /= self.scale

            #----------------closest RT (point cloud distance, RT distance)----------------
            # read RT matrices
            RT_list = []
            for rt_adr in self.rt_adrs:
                rt = np.loadtxt(rt_adr)
                RT_list.append(rt)
            assert len(RT_list) == 48
            RT_pred_rt = RT_list[rt_closest_idx]

            #--------------- POSE IMAGE ------------------------
            K = np.loadtxt(self.k_adr)
            pose_img_gt_asc = self.pointcloud_to_poseimg(ptcld_asc, K, RT_gt)
            pose_img_pred_asc = self.pointcloud_to_poseimg(ptcld_asc, K, RT_pred)
            pose_img_pred_asc_rt = self.pointcloud_to_poseimg(ptcld_asc, K, RT_pred_rt)
            pose_img_gt = self.pointcloud_to_poseimg(cornerpoints, K, RT_gt)
            pose_img_pred = self.pointcloud_to_poseimg(cornerpoints, K, RT_pred)
            pose_img_row = np.concatenate((view, pose_img_gt_asc, pose_img_pred_asc,
                pose_img_pred_asc_rt, pose_img_gt, pose_img_pred), 1)
            pose_img = np.concatenate((pose_img, pose_img_row), 0) if len(pose_img) else pose_img_row
            
        return pose_img

    ##### POSE CANDIDATE #####

    def return_closest_pose_candidate(self, cad_lab_list, b_RT_pred, dist=False):
        """find closest candidate pose from predicted Rt
        
        Args:
            cad_lab_list: {list {str}} cad filenames
            b_RT_pred: {float : (None, 3, 4)} predicted RT
        
        Returns:
            rt_closest_idx_list: {list {int}} index of closest pose (RT distance)
            rt_dist_np_list : {list {float}} distance array of candidate RT and predicted RT (RT distance)
            
        """
        rt_closest_idx_list = []
        rt_dist_list = []

        # ground truth RT list
        RT_list = []
        for rt_adr in self.rt_adrs:
            rt = np.loadtxt(rt_adr)
            RT_list.append(rt)
        RT_list = np.array(RT_list)
        assert len(RT_list) == 48

        for i, RT_pred_ in enumerate(b_RT_pred):
            # RT distance list
            rt_dist = np.mean(np.absolute(RT_list - RT_pred_), axis=(1, 2))
            rt_closest_idx = np.argmin(rt_dist)

            # append
            rt_closest_idx_list.append(rt_closest_idx)
            if dist:
                rt_dist_list.append(rt_dist)

        # manage duplicate pose
        rt_closest_idx_list = self.manage_duplicate_pose(cad_lab_list, rt_closest_idx_list)

        if dist:
            return rt_closest_idx_list, rt_dist_list
        else:
            return rt_closest_idx_list

    ##### BATCH ######

    def return_batch(self, img_lab_list, cad_lab_list, mode='train'):
        random_border = self.args.random_border_train if mode == 'train' else self.args.random_border_val
        #---------------VIEW-----------------
        b_view = []
        for img_lab in img_lab_list:
            # view image
            view_filename = os.path.join(self.dataset_dir, img_lab)
            view = cv2.imread(view_filename)
            view = self.resize_and_pad(view, random_border=random_border)
            view = np.expand_dims(view, 0)
            view = view / 255.0
            b_view = np.concatenate((b_view, view), 0) if len(b_view) else view
        #----------------CORNER POINTS-----------------------
        b_cornerpoints = []
        for cad_lab in cad_lab_list:
            cornerpoints = self.read_cornerpoints(cad_lab)
            b_cornerpoints = np.concatenate((b_cornerpoints, cornerpoints), 0) if len(b_cornerpoints) else cornerpoints
        return b_view, b_cornerpoints


    def return_testbatch(self):
        """make test batch
        
        Returns:
            b_view : {float : (batch_size, 224, 224, 3)} rendering image normalized to [0, 1]
            b_cornerpoints : {float : (batch_size, NUM_POINTS=50, 3)} corner points

        """
        pickle_path = self.args.pickle_path + '/test_set.pickle'
        with open(pickle_path, 'rb') as p:
            b_view = pickle.load(p)
            b_cornerpoints = pickle.load(p)
        return b_view, b_cornerpoints


    ###### PICKLE (TEST) ######

    def create_pickle(self, img_lab_list, cad_lab_list, mode='test'):
        """make test batch pickle
        
        Args:
            img_lab_list : {list {str}} path to cropped image file
            cad_lab_list : {list {str}} class name of cad in the image file
            mode : {'train', 'val', 'test'}
        
        Saves:
            b_view : {float : (batch_size, 224, 224, 3)} rendering image normalized to [0, 1]
            b_cornerpoints : {float : (batch_size, NUM_POINTS=50, 3)} corner points
            RT_list : list of 48 grond truth RT
        """
        #---------------VIEW-----------------
        b_view = []        
        for img_lab in img_lab_list:
            # view image
            view_filename = os.path.join(self.dataset_dir, img_lab)
            view = cv2.imread(view_filename)
            view = self.resize_and_pad(view)
            view = np.expand_dims(view, 0)
            view = view / 255.0
            b_view = np.concatenate((b_view, view), 0) if len(b_view) else view
        #----------------CORNER POINTS-----------------------
        b_cornerpoints = []
        for cad_lab in cad_lab_list:
            cornerpoints = self.read_cornerpoints(cad_lab)
            b_cornerpoints = np.concatenate((b_cornerpoints, cornerpoints), 0) if len(b_cornerpoints) else cornerpoints

        pickle_path = self.args.pickle_path + '/' + mode + '_set.pickle'
        with open(pickle_path, 'wb') as p:
            pickle.dump(b_view, p)
            pickle.dump(b_cornerpoints, p)

    






        
