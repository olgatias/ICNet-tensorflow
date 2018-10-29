import tensorflow as tf
import numpy as np
import imageio
from matplotlib import pyplot as plt

from model import ICNet
from tools import decode_labels

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
num_classes = 19

model_path = '/home/guy/ICNet-tensorflow/model/icnet_model.npy'
image_path = '/home/guy/ICNet-tensorflow/input/outdoor_1.png'




def preprocess(img):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    img = tf.expand_dims(img, dim=0)

    return img


def check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[1:3]

    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h / 32) + 1) * 32
        new_w = (int(ori_w / 32) + 1) * 32
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)

        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape









filename = image_path.split('/')[-1]
img = imageio.imread(image_path)

x = tf.placeholder(dtype=tf.float32, shape=img.shape)
img_tf = preprocess(x)
img_tf, n_shape = check_input(img_tf)









# Create network.
net = ICNet({'data': img_tf}, num_classes=num_classes, filter_scale=1)

raw_output = net.layers['conv6_cls']

# Predictions.
raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)
raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img.shape[0], img.shape[1])
raw_output_up = tf.argmax(raw_output_up, axis=3)
pred = decode_labels(raw_output_up, img.shape, num_classes)

# Init tf Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)

net.load(model_path, sess)
print('Restore from {}'.format(model_path))

preds = sess.run(pred, feed_dict={x: img})










plt.figure(1, [15, 30])
plt.subplot(121)
plt.imshow(img/255.0)
plt.axis('off')
plt.subplot(122)
plt.imshow(preds[0]/255.0)
plt.axis('off')
plt.show()













