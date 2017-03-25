import tensorflow as tf
import tensorflow.contrib.layers as tcl
import tensorflow.contrib.framework as tcf
import numpy as np
import os
import time
import random
from functools import partial

from DCGAN import DCGAN, SampleImageHook
from input_utils import get_image_paths, get_input_batch_tensor

# Newer version (newer thant 1.0.0) of LoggingTensorHook is more convenient!
# So I borrow it here.
from LoggingTensorHook import LoggingTensorHook


flags = tf.app.flags

# Training optimization params
flags.DEFINE_integer("epoch", 20, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Adam decay rate of first momentum estimate")
flags.DEFINE_float("beta2", 0.999, "Adam decay rate of second momentum estimate")
flags.DEFINE_float("l2", 1e-5, "l2 weight regularization rate. 0 means turn off.")
flags.DEFINE_string("weights_initializer", "normal", "[xavier | normal], initializer for conv weights")
flags.DEFINE_float("stddev", 0.02, "Standard deviation for random normal initializer")
flags.DEFINE_integer("train_size", np.inf, "How many training images to use. np.inf means use all.")
flags.DEFINE_integer("batch_size", 128, "The size of batch images")
flags.DEFINE_boolean("is_training", True, "True for training, False for testing")
'''
For unbalanced updating.
For each training iteration, update D `d_step` times, and then update G 'g_step' times.
'''
flags.DEFINE_integer("d_step", 1, "Number of steps to optimize D per training loop")
flags.DEFINE_integer("g_step", 1, "Number of steps to optimize G per training loop")
'''
Training strategy:
Read in new data batch and sample new random vectors (z) ...
1. per D updating step. (original paper.)
2. per training iteration.
*Note: When using strategy 2, set d_step=1, g_step=2 will produce better results.
'''
flags.DEFINE_integer("training_strategy", 1, "Which training strategy to use.") 

# GPU
flags.DEFINE_string("gpu_device", "0", "Visible GPU to use")

# Model params
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped).")
flags.DEFINE_integer("input_width", 108, "The size of image to use (will be center cropped).")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce")
flags.DEFINE_integer("output_width", 64, "The size of the output images to produce.")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color.")
flags.DEFINE_integer("z_dim", 100, "Dimension of random vector.")

# I/O
flags.DEFINE_string("dataset", "celebA", "The name of dataset")
flags.DEFINE_string("dataset_dir", "/mnt/data/roytseng/celebA", "Directory contains all the training images")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images")
flags.DEFINE_integer("input_threads", 10, "Number of input threads to read the input in parallel.")

flags.DEFINE_string("run_name", "run5", "Directory name for different runs")

flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory path to save the checkpoints")
flags.DEFINE_string("checkpoint_file", None, "Path to the checkpoint file to restore.")
flags.DEFINE_integer("ckpt_max_to_keep", None, "Max number of checkpoint files to save. `None` means keep all.")
flags.DEFINE_string("summary_dir", "logs", "Directory path to save the summary logs")
flags.DEFINE_string("sample_dir", "samples", "Directory to save the image samples")
flags.DEFINE_string("sample_fname", "image", "Filename prefix to save the image samples")
flags.DEFINE_integer("sample_size", 64, "Number of images to sample")

# Intervals
# do something every N training iterations
flags.DEFINE_integer("checkpoint_interval", 500, "name is self explained.")
flags.DEFINE_integer("summary_interval", 10, "name is self explained")
flags.DEFINE_integer("logging_interval", 10, "name is self explained")
flags.DEFINE_integer("sample_interval", 100, "name is self explained")

'''
Clarification about `Iteration` and `Step`:
    One `iteration` contains many `steps`.
    Exactly `d_step` + `g_step` many.
'''

F = flags.FLAGS


def image_reader(inputpath_queue, c_dim, crop_size, resize_size):
    '''
    This is a single input data reader component.
    We will have `F.input_threads` many of this in our 'Graph'
    to process data in parallel.
    '''
    reader = tf.WholeFileReader()
    path, value = reader.read(inputpath_queue)
    im = tf.image.decode_image(value, channels=c_dim)
    im.set_shape((None, None, c_dim))
    im = tf.to_float(im) / 127.5 - 1
    im = tf.image.resize_image_with_crop_or_pad(im, *crop_size)
    im = tf.expand_dims(im, axis=0)
    im = tf.image.resize_bicubic(im, resize_size)
    return [im, [path]]


def train():
    # The name 'global_step' is critical. DO NOT CHANGE IT!
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # Setup parameters for our image reader function 
    reader_fn = partial(image_reader, c_dim=F.c_dim, 
                        crop_size=(F.input_height, F.input_width),
                        resize_size=(F.output_height, F.output_width))
    # Get paths of training images
    image_paths = get_image_paths(F.dataset_dir, F.input_fname_pattern)
    
    if F.training_strategy == 1:
        # Make training data size divisible by (batch_size * d_step)
        if F.train_size is np.inf:
            F.train_size = len(image_paths)
        n_batch = F.train_size // F.batch_size
        F.train_size = (n_batch // F.d_step) * F.d_step * F.batch_size
        if F.train_size != len(image_paths):
            image_paths = random.sample(image_paths, F.train_size) 
    elif F.training_strategy == 2:
        # Make training data size divisible by batch size
        if F.train_size is np.inf:
            F.train_size = len(image_paths)
        n_batch = F.train_size // F.batch_size
        F.train_size = n_batch * F.batch_size
        if F.train_size != len(image_paths):
            image_paths = random.sample(image_paths, F.train_size) 
    else:
        raise ValueError('Unsupported value for training_strategy: {}'.format(F.training_strategy))
    print('Effective train_size: {}'.format(F.train_size))
    
    # Construct multiple reader component and batch the inputs
    with tf.device('/cpu:0'):
        im_gt_batch, path_batch = get_input_batch_tensor(
            reader_fn, image_paths, F.batch_size, num_epochs=F.epoch,
            num_threads=F.input_threads)
        if F.training_strategy == 2:
            # Start the 'Data Session' for processing the input data
            '''
            We need this extra session for strategy 2 only.
            Because we don't want `im_gt_batch.eval()`
            to trigger callback to the hooks defined afterwards. 
            '''
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0,
                                        visible_device_list="")
            config = tf.ConfigProto(gpu_options=gpu_options)
            sess_data = tf.Session(config=config)
            tf.local_variables_initializer().run(session=sess_data)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess_data, coord=coord)
    
    # Construct our DCGAN model, and then pass in the `input`
    if F.training_strategy == 1:
        z = tf.random_uniform((F.batch_size, F.z_dim), minval=-1, maxval=1)
        im_gt = im_gt_batch
    elif F.training_strategy == 2:
        z = tf.placeholder(tf.float32, (F.batch_size, F.z_dim), name='z')
        im_gt = tf.placeholder(tf.float32,
                               (F.batch_size, F.output_height, F.output_width, F.c_dim),
                               name='im_gt')
    input = {
        'z':        z,
        'im_gt':    im_gt,
        'sample_z': np.random.uniform(low=-1.0, high=1.0,
                                      size=(F.sample_size, F.z_dim)).astype(np.float32)
    }
    if F.weights_initializer == 'xavier':
        weights_initializer = tcl.xavier_initializer()
    elif F.weights_initializer == 'normal':
        weights_initializer = tf.random_normal_initializer(stddev=F.stddev)
    else:
        raise ValueError('Unsupported value for `weights_initializer`: {}'.format(weights_initializer))
    model = DCGAN(F, weights_initializer, tcl.l2_regularizer(scale=F.l2))
    model(input)
    
    # Show model trainable variables
    print('------------Trainable variables------------')
    for v in model.vars_train:
        print(v.name)
    print('-------------------------------------------')
    
    # Add subdirectory `run_name` under checkpoint, summary and sample directories
    F.checkpoint_dir = os.path.join(F.checkpoint_dir, F.run_name)
    F.summary_dir = os.path.join(F.summary_dir, F.run_name)
    F.sample_dir = os.path.join(F.sample_dir, F.run_name)
    
    # Setup scaffold and hooks
    '''
    Hooks: Callback objects. Perform some tasks periodically.
    Scaffold: Collect some common parameters in one place.
    '''
    step_multiplier = F.d_step + F.g_step
    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(max_to_keep=F.ckpt_max_to_keep),
        summary_op=model.summary_all)
    
    ckpt_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=F.checkpoint_dir,
        save_steps=F.checkpoint_interval * step_multiplier,
        checkpoint_basename='model',
        scaffold=scaffold)
    
    sum_hook = tf.train.SummarySaverHook(
        output_dir=F.summary_dir,
        save_steps=F.summary_interval * step_multiplier,
        scaffold=scaffold)
    
    log_hook = LoggingTensorHook(
        model.logging_tensors,
        every_n_iter=F.logging_interval * step_multiplier,
        formatter=model.logging_formatter)
    
    sample_hook = SampleImageHook(
        model,
        model.sample_img,
        os.path.join(F.sample_dir, F.sample_fname),
        every_n_iter=F.sample_interval * step_multiplier)
    
    hooks = [ckpt_hook, sum_hook, log_hook, sample_hook]
    
    '''
    Set logging level to `INFO`,
    such that we can see the logs produced by the `log_hook` we just defined above.
    '''
    tf.logging.set_verbosity(tf.logging.INFO)    
    
    # Configurations for session
    '''
    allow_growth:
        If true, the allocator does not pre-allocate the entire specified
        GPU memory region, instead starting small and growing as needed.
    '''
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=F.gpu_device)
    
    '''
    If allow_soft_placement is true, an op will be placed on CPU if
        1. there's no GPU implementation for the OP
          or
        2. no GPU devices are known or registered
          or
        3. need to co-locate with reftype input(s) which are from CPU.
    '''
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    
    '''
    Specify either `checkpoint_dir` or `checkpoint_file`, NOT both!!
    The latest checkpoint in `checkpoint_dir` will be restored if exists,
    or `checkpoint_file` will be restored.
    '''
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        config=config,
        checkpoint_dir=F.checkpoint_dir if not F.checkpoint_file else None,
        checkpoint_filename_with_path=F.checkpoint_file)
    
    # Finally, start our 'Training Session'
    sess = tf.train.MonitoredSession(session_creator=session_creator, hooks=hooks)
    try:
        if F.training_strategy == 1:
            while not sess.should_stop():
                model.train_iter(sess)
        elif F.training_strategy == 2:
            while not sess.should_stop() and not coord.should_stop():
                feed_dict = {
                    input['z']: np.random.uniform(low=-1.0, high=1.0,
                                                  size=(F.batch_size, F.z_dim)).astype(np.float32),
                    input['im_gt']: im_gt_batch.eval(session=sess_data),
                }
                model.train_iter(sess, feed_dict=feed_dict)
    except KeyboardInterrupt:
        print('Interrupt!\n [Epoch {}] [{} / {}]  [Step {}]'
              .format(model.epoch_id, model.batch_id, model.n_batch, 
                      global_step.eval(session=sess)))
    except tf.errors.OutOfRangeError as e:
        print('Epoch limit: {} reached.'.format(model.epoch_id))
        if F.training_strategy == 2:
            coord.request_stop(e)
    finally:
        if F.training_strategy == 2:
            coord.request_stop()
            coord.join(threads)
            sess_data.close()
        sess.close()
    print('Training Ended.')

            
def test():
    # Practice yourself !
    pass


if __name__ == '__main__':
    if F.is_training:
        train()
    else:
        test()
