import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import cv2
np.set_printoptions(suppress=True, precision=4)
from time import time as t
from pose_model import MODEL
from pose_dataset import Dataset
from pose_gt import *
from tabulate import tabulate
from random import shuffle
from collections import OrderedDict


class POSE_NET(object):
    """
    input : 
        manual image
        default pose CAD point cloud
    output :
        quaternion, t
    loss :
        euclidean distance between CAD point cloud transformed with (R, t) and (R_gt, t_gt)
    """

    def __init__(self, args):
        test_mode = (args.mode == 'test')
        self.args = args

        if not test_mode:
            self.eps = 1e-18
            self.hyper_list = list(map(float, args.hyper_list))

        # path
        if not test_mode:
            self.checkpoint_path = os.path.join(args.checkpoint_path, str(args.gpu), args.experiment)
            self.val_path = os.path.join(args.val_results_path, str(args.gpu), args.experiment)
            self.tensorboard_path = os.path.join(args.tensorboard_path, str(args.gpu), args.experiment)
        self.test_model_path = args.test_model_path
        self.test_results_path = args.test_results_path
        if not test_mode:
            self.ensure_path([self.checkpoint_path, self.val_path, self.tensorboard_path])
        else:
            self.ensure_path([self.test_results_path])

        # model
        if not test_mode:
            self.global_step = tf.Variable(0, trainable=False)
            self.increment_global_step = tf.assign_add(self.global_step, 1)
            self.lr = tf.train.exponential_decay(args.lr, self.global_step, args.decay_steps, args.decay_rate) + 5e-9
        
        self.bp = self.batch_provider = Dataset(self.args)
        self.is_training = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])  # view image
        self.Y = tf.placeholder(tf.float32, [None, self.batch_provider.NUM_POINTS, 3])  # point cloud
        self.Z = tf.placeholder(tf.float32, [None, 3, 4])  # RT of ground truth pose
        self.logit = self.model(self.X, self.Y, self.is_training)
        self.RT_pred, self.quat_pred, self.norms = self.logit_to_RT(self.logit)
        
        if not test_mode:
            self.ptcld_pred = self.transform_pointcloud(self.Y, self.RT_pred)
            self.ptcld_pose = self.transform_pointcloud(self.Y, self.Z)
            hyper1, hyper2, hyper3, hyper4 = self.hyper_list
            self.loss1 = hyper1 * tf.reduce_mean(self.tf_se(self.ptcld_pred - self.ptcld_pose))
            self.loss2 = hyper2 * tf.reduce_mean(tf.abs(self.tf_se(self.logit[:, :4]) - 1))
            self.loss3 = hyper3 * tf.reduce_mean(self.tf_se(self.RT_pred - self.Z, axis=[1, 2]))  # RT
            self.loss4 = hyper4 * tf.reduce_mean(self.cosine_similarity(self.RT_pred, self.Z))  # cosine similarity # seems to induce a lot of mode collapse problems
            self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss4

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='view_n_ptcld')
        self.saver = tf.train.Saver(var_list=var_list)

        if not test_mode:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=var_list)
            self.start_epoch = 0
            self.max_epoch = args.max_epoch

            # summary
            tf.summary.scalar('loss1', self.loss1)
            tf.summary.scalar('loss2', self.loss2)
            tf.summary.scalar('loss3', self.loss3)
            tf.summary.scalar('loss4', self.loss4)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('learning rate', self.lr)
            # tf.summary.image('view input', self.X)
            self.summary_op = tf.summary.merge_all()

    def tf_se(self, error, axis=-1):
        """ return squared error """
        output = tf.reduce_sum(tf.square(error), axis=axis)
        return output

    def cosine_similarity(self, RT_pred, RT_gt):
        """ Compute distance between two rotation matrices for loss
        using cosine similarity

        Args:
            RT_pred: {float32 : None x 3 x 4} predicted RT
            RT_gt: {float32 : None x 3 x 4} ground truth R
        Returns:
            1 - cosine similarity
        """
        R_pred = RT_pred[:, :3]
        R_gt = RT_gt[:, :3]
        R = tf.matmul(tf.transpose(R_pred, perm=[0, 2, 1]), R_gt)
        R_trace = tf.clip_by_value(tf.trace(R), -1.0, 3.0)
        return 1 - (0.5 * (R_trace - 1.0))

    def ensure_path(self, path_list):
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)

    def model(self, x, y, is_training):
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        logit = MODEL(x, y, is_train=is_training, regularizer=self.regularizer)
        return logit

    def logit_to_RT(self, logit):
        quaternion = logit[:, :4]
        norms = self.tf_se(quaternion)
        norms = tf.expand_dims(norms, -1)
        # quaternion = tf.div_no_nan(quaternion, norms)
        quaternion = tf.divide(quaternion, norms)
        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        R = tf.convert_to_tensor([
            [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
        ])
        R = tf.transpose(R, perm=[2, 0, 1])
        T = tf.expand_dims(logit[:, 4:], -1)
        RT = tf.concat((R, T), -1)
        return RT, quaternion, norms

    def transform_pointcloud(self, ptcld, RT):
        ones = tf.expand_dims(tf.ones_like(ptcld)[:, :, 0], -1)  # None x N x 1
        homogeneous_coordinate = tf.concat((ptcld, ones), -1)  # None x N x 4 : (x, y, z, 1)
        # None x 3xN
        coord_3d = tf.matmul(RT, tf.transpose(homogeneous_coordinate, perm=[0, 2, 1]))
        # None x Nx3 (x, y, z)
        coord_3d = tf.transpose(coord_3d, perm=[0, 2, 1])
        return coord_3d

    def create_pickles(self):
        # load labels
        test_img_labels, test_cad_labels, test_pose_labels = self.bp.shuffle_and_parse_labels(self.bp.test_labels, shuffle_labels=False)
        # duplicate pose handling
        test_pose_labels = self.bp.manage_duplicate_pose(test_cad_labels, test_pose_labels)
        # load data / make pickle
        print('Creating test set pickle')
        self.bp.create_pickle(test_img_labels, test_cad_labels, mode='test')

    def train(self):
        tic_train = t()
        time_dataloader = 0
        time_backwardpass = 0

        # load_labels
        train_labels = self.bp.train_labels
        val_labels = self.bp.val_labels

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # tensorboard
            self.writer = tf.summary.FileWriter(self.tensorboard_path)
            os.system('rm {}/events*'.format(self.tensorboard_path))
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # restore model weights
            if self.args.restore:
                if tf.train.latest_checkpoint(self.checkpoint_path) is not None:
                    print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                    print('POSE MODEL : Loading weights from %s' % tf.train.latest_checkpoint(self.checkpoint_path))
                    self.saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_path))
                    self.start_epoch = int(os.path.basename(tf.train.latest_checkpoint(self.checkpoint_path)).split('-')[-1])
                    print('Loaded')
                    print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                else:
                    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    print('No pose checkpoint found at {}'.format(self.checkpoint_path))
                    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                        
            # train
            best_score = int(self.args.best_val_accuracy * len(self.bp.val_labels))  # rt_score (maximum 48)
            print('====================================')
            print('           Train Started            ')
            print('====================================')
            for epoch in range(self.start_epoch, self.max_epoch + 1):
                tic = t()
                # load labels
                train_img_labels, train_cad_labels, train_pose_labels = self.bp.shuffle_and_parse_labels(train_labels)
                # manage duplicate pose
                train_pose_labels = self.bp.manage_duplicate_pose(train_cad_labels, train_pose_labels)

                max_batch_index = len(train_img_labels) // self.args.batch_size
                if len(train_img_labels) % self.args.batch_size != 0: max_batch_index += 1
                for batch_index in range(max_batch_index):
                    # load data
                    tic_data = t()

                    b_img_lab, b_cad_lab, b_pose_lab = self.bp.batch_labels(train_img_labels, train_cad_labels, train_pose_labels, self.args.batch_size, batch_index)
                    view, ptcld = self.bp.return_batch(b_img_lab, b_cad_lab, mode='train')
                    b_RT_pose = self.bp.return_gt_rt(b_pose_lab)
                    
                    toc_data = t()
                    time_dataloader += toc_data - tic_data
                    # train op
                    tic_trainop = t()
                    
                    fetches = [self.logit, self.quat_pred, self.norms, self.loss1, self.loss2, self.loss3, self.loss4, self.loss, self.optimizer]
                    feed_dict = {self.X: view, self.Y: ptcld, self.Z: b_RT_pose, self.is_training: True, self.global_step: epoch}
                    logit, quat_pred, norms, loss1, loss2, loss3, loss4, loss, _ = sess.run(fetches, feed_dict=feed_dict)
                    
                    toc_trainop = t()
                    time_backwardpass += toc_trainop - tic_trainop
                toc = t()
                print('epoch {} loss1 : {:.4} loss2 : {:.4} loss3 : {:.4} loss4 : {:.4} | loss : {:.4}, time per epoch: {:.4} sec, total time : {} sec'.format(
                    epoch, loss1, loss2, loss3, loss4, loss, toc - tic, int(toc - tic_train)))
                print('data loading time : {}, backward pass time : {}'.format(time_dataloader, time_backwardpass))
                print('quaternion[0] : {}, norms[0] : {}, batch_size : {}'.format(quat_pred[0], norms[0], len(quat_pred)))

                # summary
                fetches = [self.summary_op, self.increment_global_step]
                feed_dict = {self.X: view, self.Y: ptcld, self.Z: b_RT_pose, self.is_training: False}
                summary, _ = sess.run(fetches, feed_dict=feed_dict)
                self.writer.add_summary(summary, epoch)

                # validation
                if epoch % self.args.val_every == 0:
                    print('-------------- validation -------------')
                    rt_score_list = list()
                    rt_score_dict = OrderedDict()
                    for cad_name in self.bp.classes:
                        rt_score_dict[cad_name] = []

                    # initialize temp.txt
                    f = open('./temp.txt', 'w')
                    f.close()

                    # load labels
                    val_img_labels, val_cad_labels, val_pose_labels = self.bp.shuffle_and_parse_labels(val_labels, shuffle_labels=False)
                    # manage duplicate pose
                    val_pose_labels = self.bp.manage_duplicate_pose(val_cad_labels, val_pose_labels)

                    with open('./temp.txt', 'a') as f:
                        max_val_batch_index = len(val_pose_labels) // self.args.val_batch_size
                        if len(val_pose_labels) % self.args.val_batch_size != 0: max_val_batch_index += 1
                        for val_batch_index in range(max_val_batch_index):
                            # load batch
                            b_img_lab, b_cad_lab, b_pose_lab = self.bp.batch_labels(val_img_labels, val_cad_labels, val_pose_labels, self.args.val_batch_size, val_batch_index)
                            views, ptcld = self.bp.return_batch(b_img_lab, b_cad_lab, mode='val')
                            b_RT_pose = self.bp.return_gt_rt(b_pose_lab)

                            # feed forward
                            fetches = [self.ptcld_pred, self.ptcld_pose, self.RT_pred]  # TODO : quat_pred
                            feed_dict = {self.X: views, self.Y: ptcld, self.Z: b_RT_pose, self.is_training: False}
                            ptcld_pred, ptcld_pose, b_RT_pred = sess.run(fetches, feed_dict=feed_dict)

                            # closest pose
                            closest_rt_idx_list = self.bp.return_closest_pose_candidate(b_cad_lab, b_RT_pred)

                            # write results to "temp.txt"
                            for val_cad_lab, val_pose_lab, RT_gt, RT_pred, closest_rt_idx in zip(b_cad_lab, b_pose_lab, b_RT_pose, b_RT_pred, closest_rt_idx_list):
                                print('\nclass name : {},'.format(val_cad_lab), 'pose index : {}'.format(val_pose_lab), file=f)
                                print(tabulate([[RT_gt, RT_pred]], headers=['RT_gt', 'RT_pred']), file=f)
                                print('# expected answer / closest RT', file=f)
                                print(val_pose_lab, closest_rt_idx, file=f)
                                s = 1 if val_pose_lab == closest_rt_idx else 0
                                rt_score_list.append(s)

                            # save pose comparison image
                            if val_batch_index == 0:
                                pose_img_len = self.args.pose_img_len
                                pose_compare_img = self.bp.return_pose_comparison_img(b_img_lab, b_cad_lab, b_pose_lab, b_RT_pose, b_RT_pred, closest_rt_idx_list, pose_img_len)
                                cv2.imwrite(self.val_path + '/pose_val_' + str(epoch).zfill(5) + '.png', pose_compare_img)

                    # write results
                    with open(self.val_path + '/pose_val_' + str(epoch).zfill(5) + '.txt', 'w') as f:
                        for i in range(len(rt_score_list)):
                            rt_score_dict[val_cad_labels[i]].append(rt_score_list[i])
                        filelist = [sys.stdout, f]
                        for file in filelist:
                            total_score = 0
                            total_N = 0
                            for key, values in rt_score_dict.items():
                                print('<{}>'.format(key), file=file)
                                score = sum(values)
                                total_score += score
                                N = len(values)
                                total_N += N
                                try:
                                    print('rt score : {:.4}%  {}/{}'.format(score / N * 100, score, N), file=file)
                                except:
                                    pass
                            print('{:.4}% {}/{}'.format(total_score / total_N * 100, total_score, total_N), file=file)
                            print('-----------------------------', file=file)

                        # append contents of temp.txt
                        with open('./temp.txt', 'r') as g:
                            contents = g.read()
                            f.write(contents)

                    # save model
                    if total_score > best_score:
                        if self.args.save:
                            try:
                                self.saver.save(sess, self.checkpoint_path + '/saved_model', global_step=epoch)
                                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                                print('Saved model of epoch {}'.format(epoch))
                                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                            except:
                                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                                print('   Failed saving model ㅜㅜ   ')
                                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        best_score = np.sum(rt_score_list)

        print('====================================')
        print('           Train Completed          ')
        print('====================================')

    def test(self):
        tic = t()
        # load labels
        test_labels = self.bp.test_labels
        test_img_labels, test_cad_labels, test_pose_labels = self.bp.shuffle_and_parse_labels(test_labels, shuffle_labels=False)
        # duplicate pose handling
        test_pose_labels = self.bp.manage_duplicate_pose(test_cad_labels, test_pose_labels)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # restore model weights
            if tf.train.latest_checkpoint(self.test_model_path) is not None:
                print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
                print('POSE MODEL : Loading weights from %s' % tf.train.latest_checkpoint(self.test_model_path))
                self.saver.restore(sess, tf.train.latest_checkpoint(self.test_model_path))
                print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
            else:
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('No pose checkpoint found at {}'.format(self.test_model_path))
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                sys.exit()
            toc = t()
            print('model loading time :', toc - tic)
            tic = t()
            views, cornerpoints = self.bp.return_testbatch()
            toc = t()
            print('data loading time :', toc - tic)
            tic = t()

            # feed forward
            fetches = [self.RT_pred]
            feed_dict = {self.X: views, self.Y: cornerpoints, self.is_training: False}
            [b_RT_pred] = sess.run(fetches, feed_dict=feed_dict)
            toc = t()
            print('feed forward time :', toc - tic)
            tic = t()

            # closest pose
            rt_idx_list = self.bp.return_closest_pose_candidate(test_cad_labels, b_RT_pred, dist=False)
            # calculate scores
            rt_score_list = []
            for closest_rt_idx, gt_pose_idx in zip(rt_idx_list, test_pose_labels):
                s = 1 if closest_rt_idx == gt_pose_idx else 0
                rt_score_list.append(s)
            if self.args.test_simple:
                total_score = sum(rt_score_list)
                total_N = len(rt_score_list)
                print('{:.4}% {}/{}'.format(total_score / total_N * 100, total_score, total_N))
                print('-----------------------------')
            else:
                if self.args.save_test_imgs:
                    count = 0
                    LABEL_TO_POSE = {v: k for k, v in POSE_TO_LABEL.items()}
                    for input_img, closest_rt_idx, gt_pose_idx, cad_name in zip(views, rt_idx_list, test_pose_labels, test_cad_labels):
                        # save image
                        plt.clf()
                        fig, ax = plt.subplots(1, 2, sharey=True)
                        input_img = (input_img * 255).astype(np.uint8)
                        class_index = self.bp.classes.index(cad_name)
                        pred_pose_img = cv2.imread(self.bp.rendering_adrs[class_index][closest_rt_idx])
                        ax[0].imshow(input_img)
                        ax[0].set_title('{}\n{}'.format(cad_name, LABEL_TO_POSE[gt_pose_idx]))
                        ax[1].imshow(pred_pose_img)
                        ax[1].set_title('pred pose : {}'.format(LABEL_TO_POSE[closest_rt_idx]))
                        if closest_rt_idx == gt_pose_idx:
                            plt.savefig(self.test_results_path + '/correct_' + str(count).zfill(3) + '.png')
                        else:
                            plt.savefig(self.test_results_path + '/wrong_' + str(count).zfill(3) + '.png')
                        count += 1
                        plt.close()
                rt_score_dict = OrderedDict()
                for cad_name in self.bp.classes:
                    rt_score_dict[cad_name] = []
                for i in range(len(rt_score_list)):
                    rt_score_dict[test_cad_labels[i]].append(rt_score_list[i])
                filelist = [sys.stdout]
                for file in filelist:
                    total_score = 0
                    total_N = 0
                    for key, values in rt_score_dict.items():
                        print('<{}>'.format(key), file=file)
                        score = sum(values)
                        total_score += score
                        N = len(values)
                        total_N += N
                        print('rt score : {:.4}%  {}/{}'.format(score / N * 100, score, N), file=file)
                    print('{:.4}% {}/{}'.format(total_score / total_N * 100, total_score, total_N), file=file)
                    print('-----------------------------', file=file)
            toc = t()
            print('find closest pose / score calculation time :', toc - tic)
