import tensorflow as tf # tensorflow takes 0.8 sec to import. other time is negligible
import os

def define_args():
    FLAGS = tf.app.flags.FLAGS

    # directories 
    tf.app.flags.DEFINE_string('krt_path', './input/stefan/views/KRT', """krt path""")
    tf.app.flags.DEFINE_string('views_path', './input/stefan/views/VIEWS', """views path""")
    tf.app.flags.DEFINE_string('cad_path', './input/stefan/cad', """cad path""")
    tf.app.flags.DEFINE_string('point_cloud_path', './input/stefan/cad/point_cloud', """point cloud path""")
    tf.app.flags.DEFINE_string('dataset_path', '../Pose_Dataset', """directory to pose_datset""")
    tf.app.flags.DEFINE_string('test_imgs_path', './input/stefan/detection', """cropped manual images path""")
    tf.app.flags.DEFINE_string('renderings_path', './input/stefan/views/VIEWS', """renderings path""")
    tf.app.flags.DEFINE_string('checkpoint_path', './checkpoints/', """checkpoint directory""")
    tf.app.flags.DEFINE_string('test_model_path', './test_model', """checkpoint directory""")
    tf.app.flags.DEFINE_string('tensorboard_path', './tb', """tensorboard directory""")
    tf.app.flags.DEFINE_string('background_path', './background', """background imgs for data aug directory""")
    tf.app.flags.DEFINE_string('val_results_path', './validation', """validation results path""")
    tf.app.flags.DEFINE_string('test_results_path', './test_results', """test results path""")
    tf.app.flags.DEFINE_string('experiment', 'experiment', """experiment name""")    
    tf.app.flags.DEFINE_string('pickle_path', './pickle', """test results path""")
    
    # training
    tf.app.flags.DEFINE_integer('batch_size', 2, """batch size""")
    tf.app.flags.DEFINE_integer('val_batch_size', 8, """validation batch size""")
    tf.app.flags.DEFINE_integer('train_size', 0, """for debugging""")
    tf.app.flags.DEFINE_integer('pose_img_len', 8, """number of rows in validation pose image""")
    tf.app.flags.DEFINE_string('img_size', '224,224', """expected input image size, HEIGHT, WIDTH""")
    tf.app.flags.DEFINE_integer('view_num', 48, """number of views for each model""")
    tf.app.flags.DEFINE_integer('scale', 1000, """cad model scale down factor""")
    tf.app.flags.DEFINE_integer('ptcld_len', 50, """number of points used from point cloud (maximum 84 for 'chair_part_1')""")
    tf.app.flags.DEFINE_spaceseplist('hyper_list', '1 0.01 1 1', 'string divided by spaces')
    tf.app.flags.DEFINE_bool('restore', True, """whether to continue training""")
    tf.app.flags.DEFINE_float('best_val_accuracy', 0, """best start validation accuracy for continued training""")
    tf.app.flags.DEFINE_bool('save', True, """whether to save checkpoints""")
    tf.app.flags.DEFINE_string('mode', 'train', """{train, test}""")
    tf.app.flags.DEFINE_string('gpu', '1', """gpu number to be used""")
    tf.app.flags.DEFINE_integer('max_epoch', 1000, """maximum number of epoch""")
    tf.app.flags.DEFINE_float('lr', 1e-4, """initial learning rate""")
    tf.app.flags.DEFINE_integer('val_every', 1, """validation epoch stride""")
    tf.app.flags.DEFINE_integer('test_every', 1000, """test epoch stride""")
    tf.app.flags.DEFINE_integer('decay_steps', 50 , """lr = decay * lr period epoch""")
    tf.app.flags.DEFINE_float('decay_rate', 0.95, """lr decay rate""")
    tf.app.flags.DEFINE_bool('random_border_train', True, """apply random border to train images""")
    tf.app.flags.DEFINE_bool('random_border_val', True, """apply random border to validation images""")

    # test
    tf.app.flags.DEFINE_integer('num_test', 10, """number of whole images""")
    tf.app.flags.DEFINE_bool('test_simple', True, """only print accuracy result""")
    tf.app.flags.DEFINE_bool('save_test_imgs', True, """save test images, when test_simple is False""")

    ensure_paths(FLAGS)
    return FLAGS


def ensure_paths(args):
    """create directories for path arguments in args
    argument name must include 'path' or 'dir'
    """
    path_list = []
    if 'argparse' in str(type(args)):
        for arg in vars(args):
            if 'path' in arg or 'dir' in arg:
                path_list.append(getattr(args, arg))
    elif 'tensorflow' in str(type(args)):
        for arg in args.flag_values_dict():
            if 'path' in arg or 'dir' in arg:
                path_list.append(args.flag_values_dict()[arg])
    # print(path_list)
    for path in path_list:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                pass


if __name__ == '__main__':
    FLAGS = define_args()
