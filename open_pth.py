import tensorflow as tf
import torch
def convert(bin_path, ckptpath):
    with tf.compat.v1.Session() as sess:
        for var_name, value in torch.load(bin_path, map_location='cpu').items():
            # print(var_name)
            tf.Variable(initial_value=value, name=var_name)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer0)
            saver.save(sess, ckpt_path)
bin_path = './pretrained/aimnet_pretrained_duts.pth'
ckpt_path = '../pretrained/aimnet_pretrained_duts.ckpt'
convert(bin_path, ckpt_path)
