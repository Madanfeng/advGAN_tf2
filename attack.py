import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '6'


adv_img_path = "./adv_demo.jpg"
save_img_path = "./demo.jpg"

# load generator
gen_path = "./weights/generator/25.h5"
generator = tf.keras.models.load_model(gen_path, compile=False)
gen_model = Model(inputs=generator.input, outputs=generator.layers[-1].output)
# gen_model.summary()

# load model
model_path = "/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/model/model.stage3.03-0.8788.v2_320_tf2.hdf5"
model = tf.keras.models.load_model(model_path, compile=False)
tmodel = Model(inputs=model.input, outputs=model.layers[-1].output)

# load img
img_path = "/data/project/xing/porn_data_set/multi_label_porn_misclassify_20200408/2eabea21d7ebf83f0ec9dd32bd4bd933.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (320, 320))
cv2.imwrite(save_img_path, img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img[np.newaxis, :] / 255. - 0.5

# produce adv img
perturb = gen_model(img)
perturb = tf.clip_by_value(perturb, -0.5, 0.5)
adv_img = img + perturb
adv_img = tf.clip_by_value(adv_img, -0.5, 0.5)

# pred
pred_adv = tmodel(adv_img)
print("adv_img:")
print(pred_adv.numpy())
pred = tmodel(img)
print("img:")
print(pred.numpy())

# save adv img
adv_img = (adv_img[0].numpy() + .5) * 255.
adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(adv_img_path, adv_img)

