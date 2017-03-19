import tensorflow as tf
import tensorflow.contrib.layers as tcl
import tensorflow.contrib.framework as tcf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
import os
import math

from models import Discriminator, Generator
from image_utils import save_images

class DCGAN():
    def __init__(self, F, weights_initializer=tcl.xavier_initializer(), regularizer=None):
        '''
        Args:
          F: Contain the parameters(FLAGS) for setting the model
        '''
        self.F = F
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer
        self.D = Discriminator(is_training=F.is_training,
                               weights_initializer=weights_initializer)
        self.G = Generator(F.output_height, F.output_width,
                           is_training=F.is_training,
                           weights_initializer=weights_initializer)
        if F.is_training:
            self.epoch_id = 1
            self.batch_id = 1
            
            # We expect no remainders, to make the calculated `batch_id` precise
            self.n_batch = F.train_size // F.batch_size  
            
            # width param to format verbose information
            self.batch_width = len(str(self.n_batch))
            if F.training_strategy == 1:
                self.step_width = len(str(F.epoch * self.n_batch * (1 + F.g_step / F.d_step)))
            elif F.training_strategy == 2:
                self.step_width = len(str(F.epoch * self.n_batch * (F.d_step + F.g_step)))
    
    def __call__(self, input, model_name=None, reuse=False):
        if not model_name:
            model_name = self.__class__.__name__
        with tf.variable_scope(model_name, reuse=reuse) as self.model_scope:
            self.input = input
            output = self.setup_model(input)
        return output
    
    @property
    def vars_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                 scope=self.model_scope.name)
    
    @property
    def vars_train(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                 scope=self.model_scope.name)
    
    @property
    def logging_tensors(self):
        return {'step': tf.train.get_global_step(),
                'd_loss': self.d_loss,
                'g_loss': self.g_loss}
    
    def logging_formatter(self, results):
        '''
        Display verbose training information on the terminal.
        Use `tf.train.LoggingTensorHook` to implement this functionality.
        '''
        message = ('[Epoch {:02d}] [{:{batch_width}d} / {:d}] [Step {:{step_width}d}]'
                   '  D_loss: {:.5f}, G_loss: {:.5f}'
                   .format(self.epoch_id, self.batch_id, self.n_batch, results['step'],
                           results['d_loss'], results['g_loss'], 
                           batch_width=self.batch_width, step_width=self.step_width))
        return message
    
    def setup_model(self, input):
        '''
        Args:
          input: a dictionary contains 'z', 'im_gt', sample_z
        '''
        F = self.F
        #########################
        # (1) Define main model #
        #########################
        g_out = self.G(input['z'])
        d_out_real, d_logit_real = self.D(input['im_gt'])
        d_out_fake, d_logit_fake = self.D(g_out, reuse=True)
        self.output = g_out
        
        self.sample_img = self.G(input['sample_z'], reuse=True)
        
        ###################
        # (2) Define loss #
        ###################
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_out_real),
                                                    logits=d_logit_real))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_out_fake),
                                                    logits=d_logit_fake))
        g_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_out_fake),
                                                    logits=d_logit_fake))
        
        d_reg_loss = tcl.apply_regularization(self.regularizer, weights_list=self.D.vars_train)
        g_reg_loss = tcl.apply_regularization(self.regularizer, weights_list=self.G.vars_train)
        
        self.d_loss = d_loss_real + d_loss_fake + d_reg_loss
        self.g_loss = g_loss_fake + g_reg_loss
        
        ########################
        # (3) Define optimizer #
        ########################
        global_step = tf.train.get_global_step()
        d_optimizer = tf.train.AdamOptimizer(learning_rate=F.learning_rate, beta1=F.beta1, beta2=F.beta2)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=F.learning_rate, beta1=F.beta1, beta2=F.beta2)
        self.d_train_op = tcl.optimize_loss(
            loss=self.d_loss, optimizer=d_optimizer, learning_rate=F.learning_rate,
            variables=self.D.vars_train, global_step=global_step, name='d_optim')
        self.g_train_op = tcl.optimize_loss(
            loss=self.g_loss, optimizer=g_optimizer, learning_rate=F.learning_rate,
            variables=self.G.vars_train, global_step=global_step, name='g_optim')
        
        ######################
        # (4) Define summary #
        ######################
        # scalar summary
        tf.summary.scalar('d_loss_real', d_loss_real)
        tf.summary.scalar('d_loss_fake', d_loss_fake)
        tf.summary.scalar('g_loss_fake', g_loss_fake)
        # histogram summary
        tf.summary.histogram('z', input['z'])
        tf.summary.histogram('d_out_real', d_out_real)
        tf.summary.histogram('d_out_fake', d_out_fake)
        # image summary
        tf.summary.image('generated', g_out, max_outputs=3)
        tf.summary.image('real', input['im_gt'], max_outputs=3)
        # merge all summary operations to a single operation
        self.summary_all = tf.summary.merge_all()
         
        return self.output
    
    def train_iter(self, sess, feed_dict=None):
        F = self.F
        for i in range(F.d_step):
            _= sess.run([self.d_train_op], feed_dict=feed_dict)
            
            if F.training_strategy == 1:
                # increase batch_id and maybe epoch_id
                self.batch_id += 1
                if self.batch_id > self.n_batch:
                    self.batch_id = 1
                    self.epoch_id += 1
        
        for i in range(F.g_step):
            _ = sess.run([self.g_train_op], feed_dict=feed_dict)
            
        if F.training_strategy == 2:
            # increase batch_id and maybe epoch_id
            self.batch_id += 1
            if self.batch_id > self.n_batch:
                self.batch_id = 1
                self.epoch_id += 1


class SampleImageHook(tf.train.SessionRunHook):
    def __init__(self, model, sample_img, img_path, every_n_iter=None, every_n_secs=None):
        '''
        Args:
          model : In order to retrieve `model.epoch_id` and `model.batch_id` for naming.
          sample_img : `Tensor`, sample images to save.
          img_path: 'String', path containing the directory and filename prefix
          every_n_iter: `int`, save the sample images every N local steps.
          every_n_secs: `int` or `float`, save sample images every N seconds. 
                Exactly one of `every_n_iter` and `every_n_secs` should be provided.
        '''
        self.model = model
        self.sample_img = sample_img
        self.img_path = img_path
        # Calculate appropriate grid size automatically
        h = math.sqrt(sample_img.get_shape().as_list()[0])
        w = math.ceil(h)
        self.grid_size = (int(h), w)
        self._timer = SecondOrStepTimer(every_secs=every_n_secs,
                                        every_steps=every_n_iter)

    def begin(self):
        # Make the dir if not exist
        img_dir = os.path.dirname(self.img_path)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # Counter for run iterations
        self._iter_count = 0

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            requests = {'sample_img': self.sample_img,
                        'g_out': self.model.output,}
                        # 'gt_img': self.model.input['im_gt']}
            return tf.train.SessionRunArgs(requests)
        else:
            return None

    def after_run(self,  run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._timer.update_last_triggered_step(self._iter_count)
            # Save sample images, visualizing the current training results
            save_images(self.img_path+'_%02d_%04d.jpg' % (self.model.epoch_id, self.model.batch_id),
                        run_values.results['sample_img'],
                        self.grid_size)
            # save_images(self.img_path+'_%02d_%04d_out.jpg' % (self.model.epoch_id, self.model.batch_id),
            #             run_values.results['g_out'],
            #             self.grid_size)
            ## For checking. Save groundtruth (natuarl) training images.
            # save_images(self.img_path+'_%02d_%04d_gt.jpg' % (self.model.epoch_id, self.model.batch_id),
            #             run_values.results['gt_img'][:64],
            #             self.grid_size)
        self._iter_count += 1
            