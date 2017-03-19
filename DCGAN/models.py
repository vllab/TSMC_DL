import tensorflow as tf
import tensorflow.contrib.layers as tcl
import tensorflow.contrib.framework as tcf


# manually define the activation function: LeakyRelu (tf doesn't provide one)
def lrelu(input, alpha=0.2, max_value=None, name=None):
    with tf.name_scope(name, 'lrelu', [input]) as scope:
        input = tf.convert_to_tensor(input, name="x")
        x = tf.nn.relu(input)
        if max_value:
            x = tf.clip_by_value(x, 0., max_value)
        if alpha:
            x -= alpha * tf.nn.relu(-input)
    return x


class Discriminator():
    def __init__(self, depth=4, f_num_init=64, k_size=5, stride=2, is_training=True):
        self.depth = depth              # how many conv layers to use
        self.f_num_init = f_num_init    # initial conv filter num of discriminator
        self.k_size = k_size            # conv kernel size
        self.stride = stride            # conv stride size
        self.is_training = is_training  # batch normalization has different behaviors in training and testing

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

    def setup_model(self, input):
        # Parameters for batch normalization
        bn_params = {'decay': 0.9,                 # default is 0.999
                     'center': True,               # default is True
                     'scale': True,               # default is False
                     'updates_collections': None,  # default to tf.GraphKeys.UPDATE_OPS
                     'is_training': self.is_training}
        # conv filer number
        f_num = self.f_num_init
        
        x = input
        # setup common param here, avoid redundancy
        with tcf.arg_scope([tcl.conv2d],
                           kernel_size=self.k_size,
                           stride=self.stride, 
                           activation_fn=lrelu):
            # convolution layer
            x = tcl.conv2d(x, f_num)
            for i in range(1, self.depth):
                f_num *= 2
                x = tcl.conv2d(x, f_num,
                               normalizer_fn=tcl.batch_norm,
                               normalizer_params=bn_params)
        # reshape matrix to a vector
        x = tcl.flatten(x)
        # linear (fully connected) layer: W*x + b
        self.logit = tcl.linear(x, 1)
        self.output = tf.nn.sigmoid(self.logit)
        
        return self.output, self.logit


class Generator():
    def __init__(self, h_out, w_out, depth=4, f_num_init=512, k_size=5, stride=2, c_dim=3, is_training=True):
        self.h_out = h_out     # height of output image
        self.w_out = w_out     # width of output image
        self.c_dim = c_dim     # number of color channel
        self.depth = depth
        self.k_size = k_size
        self.stride = stride
        self.f_num_init = f_num_init
        self.is_training = is_training

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

    def setup_model(self, input):
        bn_params = {'decay': 0.9,                 # default is 0.999
                     'center': True,               # default is True
                     'scale': True,               # default is False
                     'updates_collections': None,  # if None use tf.GraphKeys.UPDATE_OPS
                     'is_training': self.is_training}
        
        # calculate the input size for the first conv2d_tranpose layer
        f_num = self.f_num_init                  # num of filter
        h_num = self.h_out // 2 ** self.depth    # height
        w_num = self.w_out // 2 ** self.depth    # width
        
        # input is the random vector z
        x = input
        
        # project z to higher dimension (h_num * w_num * f_num)
        x = tcl.fully_connected(
            x,
            h_num * w_num * f_num,
            normalizer_fn=tcl.batch_norm,
            normalizer_params=bn_params,
            activation_fn=tf.nn.relu)
            #weights_initializer=tf.random_normal_initializer(stddev=0.02))
        
        # and reshape it to a 4-D tensor
        x = tf.reshape(x, [-1, h_num, w_num, f_num])
        
        with tcf.arg_scope([tcl.conv2d_transpose],
                           kernel_size=self.k_size,
                           stride=self.stride,
                           normalizer_fn=tcl.batch_norm,
                           normalizer_params=bn_params,
                           activation_fn=tf.nn.relu):
                           #weights_initializer=tf.random_normal_initializer(stddev=0.02)):
            for i in range(self.depth-1):
                f_num //= 2
                x = tcl.conv2d_transpose(x, f_num)
            
            self.output = tcl.conv2d_transpose(
                x, self.c_dim, normalizer_fn=None, activation_fn=tf.nn.tanh)

        return self.output
