import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from config import cfg
from data import Dataset, dataset
from models import Generator, Discriminator, target_model

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS


def check_dir(dir):
    if not os.path.exists(dir):
        print("Make", dir)
        os.makedirs(dir)


# loss function to encourage misclassification after perturbation
def adv_loss(preds, labels, is_targeted):
    confidence = cfg.CONFIDENCE
    real = tf.reduce_sum(labels * preds, 1)
    other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
    if is_targeted:
        return tf.reduce_sum(tf.maximum(0.0, other - real + confidence))
    return tf.reduce_sum(tf.maximum(0.0, real - other + confidence))


# loss function to influence the perturbation to be as close to 0 as possible
def perturb_loss(preds, thresh=0.3):
    zeros = tf.zeros((tf.shape(preds)[0]))
    return tf.reduce_mean(tf.maximum(zeros, tf.norm(tf.reshape(preds, (tf.shape(preds)[0], -1)), axis=1) - thresh))


class AdvGan:
    def __init__(self, train_ds, test_ds, tmodel):
        self.thresh = cfg.THRESH
        self.smooth = cfg.SMOOTH
        self.is_targeted = cfg.TARGETED

        self.tmodel = tmodel

        # ds
        self.train_data, self.train_steps_per_epoch = train_ds
        self.test_data, self.test_steps_per_epoch = test_ds

        self.discriminator = Discriminator()
        self.generator = Generator()

        # define optimizers
        self.d_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)
        self.g_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, images, targets_tensor_onehot):
        with tf.GradientTape() as disc_tape, tf.GradientTape(persistent=True) as gen_tape:
            disc_tape.watch(self.discriminator.trainable_variables)
            gen_tape.watch(self.generator.trainable_variables)

            # generate perturbation, add to original input image(s)
            perturb = tf.clip_by_value(self.generator(images, training=True), -self.thresh, self.thresh)
            images_perturbed = perturb + images
            images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)

            # pass real and perturbed image to discriminator and the target model
            d_real_logits = self.discriminator(images, training=True)
            d_real_probs = tf.nn.sigmoid(d_real_logits)

            d_fake_logits = self.discriminator(images_perturbed, training=True)
            d_fake_probs = tf.nn.sigmoid(d_fake_logits)

            # pass real and perturbed images to the model we are trying to fool
            # f_real_logits, f_real_probs = self.tmodel.predict_logits(images)
            f_fake_logits, f_fake_probs = self.tmodel.predict_logits(images_perturbed)

            # generate labels for discriminator (optionally smooth labels for stability)
            d_labels_real = tf.ones_like(d_real_probs) * (1 - self.smooth)
            d_labels_fake = tf.zeros_like(d_fake_probs)

            # discriminator loss
            d_loss_real = self.loss_object(d_labels_real, d_real_probs)
            d_loss_fake = self.loss_object(d_labels_fake, d_fake_probs)
            # d_loss_real = tf.keras.losses.MSE(y_pred=d_real_probs, y_true=d_labels_real)
            # d_loss_fake = tf.keras.losses.MSE(y_pred=d_fake_probs, y_true=d_labels_fake)
            d_loss = d_loss_real + d_loss_fake

            # generator loss
            g_loss_fake = self.loss_object(tf.ones_like(d_fake_probs), d_fake_probs)
            # g_loss_fake = tf.losses.MSE(y_pred=d_fake_probs, y_true=tf.ones_like(d_fake_probs))

            # perturbation loss (minimize overall perturbation)
            l_perturb = perturb_loss(perturb, self.thresh)

            # adversarial loss (encourage misclassification)
            l_adv = adv_loss(f_fake_logits, targets_tensor_onehot, self.is_targeted)

            # weights for generator loss function
            alpha = 5.0
            beta = 1.0
            g_loss = l_adv + alpha * g_loss_fake + beta * l_perturb

        # train the discriminator
        for _ in range(1):
            disc_grad = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

        # train the generator
        for _ in range(4):
            gen_grad = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        return g_loss, d_loss

    def fit(self, epoch):
        # if epoch == int(cfg.EPOCHS * 0.5):
        #     self.d_opt = tf.keras.optimizers.Adam(4e-5, beta_1=0.5)
        #     self.g_opt = tf.keras.optimizers.Adam(4e-5, beta_1=0.5)
        # if epoch == int(cfg.EPOCHS * 0.8):
        #     self.d_opt = tf.keras.optimizers.Adam(4e-6, beta_1=0.5)
        #     self.g_opt = tf.keras.optimizers.Adam(4e-6, beta_1=0.5)

        # train advgan
        train_it = iter(self.train_data.take(self.train_steps_per_epoch + 1))
        next(train_it)
        pbar = tqdm(train_it)
        for images, labels, targets, paths in pbar:
            # convert to one-hot
            targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
            targets_tensor_onehot = tf.one_hot(targets_tensor, depth=cfg.NUM_CLASS)

            g_loss, d_loss = self.train_step(images, targets_tensor_onehot)

            pbar.set_description("epoch: %d, g_loss: %.8f, d_loss: %.8f" % (epoch, g_loss.numpy(), d_loss.numpy()))

        # save model
        if epoch % 5 == 0:
            print("Save model...")
            self.discriminator.save(cfg.DISC_SAVE_DIR+str(epoch)+".h5", save_format='h5')
            self.generator.save(cfg.GEN_SAVE_DIR+str(epoch)+".h5", save_format='h5')

    def test(self):
        # test
        suc_num = 0
        sum_num = 0
        p_dis = 0
        sum_prob = 0

        test_it = iter(self.test_data.take(self.test_steps_per_epoch + 1))
        next(test_it)
        for images, labels, targets, paths in test_it:

            perturbs = tf.clip_by_value(self.generator(images, training=False), -self.thresh, self.thresh)
            images_perturbed = perturbs + images
            images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)

            probs = self.tmodel.predict_softmax(images_perturbed)

            for i, prob in enumerate(probs):
                sum_num += 1

                ind = tf.argmax(prob)
                if ind == targets[i].numpy(): suc_num += 1
                if cfg.TARGETED:
                    sum_prob += prob.numpy()[np.int(targets[i])]
                elif not cfg.TARGETED and ind == targets[i]:
                    sum_prob += (1. - prob.numpy()[np.int(targets[i])]) / (cfg.NUM_CLASS - 1 + 1e-7)
                else:
                    sum_prob += prob.numpy()[ind]

                p_dis += np.sum((perturbs[i] ** 2) ** .5)

        acc = suc_num / sum_num
        if not cfg.TARGETED:
            acc = 1.0 - acc
        dis = p_dis / sum_num
        ave_prob = sum_prob / sum_num
        print("Attack Success Rate:", acc, "Average Confidence:", ave_prob, "Perturb Distance:", dis)

# train function advGAN
# def AdvGAN(train_data, test_data, train_steps_per_epoch, test_steps_per_epoch, tmodel):
#     thresh = cfg.THRESH
#     smooth = cfg.SMOOTH
#     is_targeted = cfg.TARGETED
#
#     discriminator = Discriminator()
#     generator = Generator()
#
#     # define optimizers
#     d_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)
#     g_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)
#
#     loss_object = tf.keras.losses.BinaryCrossentropy()
#
#     print("Train GAN...")
#     for epoch in range(cfg.EPOCHS):
#
#         if epoch == int(cfg.EPOCHS * 0.5):
#             d_opt = tf.keras.optimizers.Adam(4e-5, beta_1=0.5)
#             g_opt = tf.keras.optimizers.Adam(4e-5, beta_1=0.5)
#         if epoch == int(cfg.EPOCHS * 0.8):
#             d_opt = tf.keras.optimizers.Adam(4e-6, beta_1=0.5)
#             g_opt = tf.keras.optimizers.Adam(4e-6, beta_1=0.5)
#
#         it = iter(train_data.take(train_steps_per_epoch + 1))
#         next(it)
#         pbar = tqdm(it)
#         for images, labels, targets, paths in pbar:
#             # convert to one-hot
#             targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
#             targets_tensor_onehot = tf.one_hot(targets_tensor, depth=cfg.NUM_CLASS)
#
#             # LOSS DEFINITIONS
#             with tf.GradientTape() as disc_tape, tf.GradientTape(persistent=True) as gen_tape:
#                 disc_tape.watch(discriminator.trainable_variables)
#                 gen_tape.watch(generator.trainable_variables)
#
#                 # generate perturbation, add to original input image(s)
#                 perturb = tf.clip_by_value(generator(images, training=True), -thresh, thresh)
#                 images_perturbed = perturb + images
#                 images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)
#
#                 # pass real and perturbed image to discriminator and the target model
#                 d_real_logits = discriminator(images, training=True)
#                 d_real_probs = tf.nn.sigmoid(d_real_logits)
#
#                 d_fake_logits = discriminator(images_perturbed, training=True)
#                 d_fake_probs = tf.nn.sigmoid(d_fake_logits)
#
#                 # pass real and perturbed images to the model we are trying to fool
#                 # f_real_logits, f_real_probs = tmodel.predict_logits(images)
#                 f_fake_logits, f_fake_probs = tmodel.predict_logits(images_perturbed)
#
#                 # generate labels for discriminator (optionally smooth labels for stability)
#                 d_labels_real = tf.ones_like(d_real_probs) * (1 - smooth)
#                 d_labels_fake = tf.zeros_like(d_fake_probs)
#
#                 # discriminator loss
#                 d_loss_real = loss_object(d_labels_real, d_real_probs)
#                 d_loss_fake = loss_object(d_labels_fake, d_fake_probs)
#                 # d_loss_real = tf.keras.losses.MSE(y_pred=d_real_probs, y_true=d_labels_real)
#                 # d_loss_fake = tf.keras.losses.MSE(y_pred=d_fake_probs, y_true=d_labels_fake)
#                 d_loss = d_loss_real + d_loss_fake
#
#                 # generator loss
#                 g_loss_fake = loss_object(tf.ones_like(d_fake_probs), d_fake_probs)
#                 # g_loss_fake = tf.losses.MSE(y_pred=d_fake_probs, y_true=tf.ones_like(d_fake_probs))
#
#                 # perturbation loss (minimize overall perturbation)
#                 l_perturb = perturb_loss(perturb, thresh)
#
#                 # adversarial loss (encourage misclassification)
#                 l_adv = adv_loss(f_fake_logits, targets_tensor_onehot, is_targeted)
#
#                 # weights for generator loss function
#                 alpha = 1.0
#                 beta = 5.0
#                 g_loss = l_adv + alpha * g_loss_fake + beta * l_perturb
#
#             # train the discriminator
#             for _ in range(1):
#                 disc_grad = disc_tape.gradient(d_loss, discriminator.trainable_variables)
#                 d_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))
#
#             # train the generator
#             for _ in range(4):
#                 gen_grad = gen_tape.gradient(g_loss, generator.trainable_variables)
#                 g_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
#
#             pbar.set_description("epoch: %d, gen_loss: %.8f, d_loss: %.8f" % (epoch, g_loss.numpy(), d_loss.numpy()))
#
#         # test
#         suc_num = 0
#         sum_num = 0
#         p_dis = 0
#         sum_prob = 0
#
#         test_it = iter(test_data.take(test_steps_per_epoch + 1))
#         next(test_it)
#         for images, labels, targets, paths in test_it:
#
#             perturbs = tf.clip_by_value(generator(images, training=False), -thresh, thresh)
#             images_perturbed = perturbs + images
#             images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)
#
#             probs = tmodel.predict_softmax(images_perturbed)
#
#             for i, prob in enumerate(probs):
#                 sum_num += 1
#
#                 ind = tf.argmax(prob)
#                 if ind == targets[i].numpy(): suc_num += 1
#                 if cfg.TARGETED:
#                     sum_prob += prob.numpy()[np.int(targets[i])]
#                 elif not cfg.TARGETED and ind == targets[i]:
#                     sum_prob += (1. - prob.numpy()[np.int(targets[i])])/(cfg.NUM_CLASS-1 + 1e-7)
#                 else:
#                     sum_prob += prob.numpy()[ind]
#
#                 p_dis += np.sum((perturbs[i] ** 2) ** .5)
#
#         acc = suc_num / sum_num
#         if not cfg.TARGETED:
#             acc = 1.0 - acc
#         dis = p_dis / sum_num
#         ave_prob = sum_prob / sum_num
#         print("Attack Success Rate:", acc, "Average Confidence:", ave_prob, "Perturb Distance:", dis)
#
#         # print("Save last model...")
#         # discriminator.save(cfg.DISC_SAVE_DIR + "lsat.h5", save_format='h5')
#         # generator.save(cfg.GEN_SAVE_DIR + "last.h5", save_format='h5')
#
#         if epoch % 1 == 0:
#             print("Save model...")
#             discriminator.save(cfg.DISC_SAVE_DIR+str(epoch)+".h5", save_format='h5')
#             generator.save(cfg.GEN_SAVE_DIR+str(epoch)+".h5", save_format='h5')


if __name__ == '__main__':
    # load myself dataset
    # train_data = Dataset(istrain=True)
    # test_data = Dataset(istrain=False)

    # load tf.data.Dataset, more efficiently
    train_data, train_num = dataset(istrain=True)
    test_data, test_num = dataset(istrain=False)

    train_steps_per_epoch = int(train_num / cfg.BATCH_SIZE)
    test_steps_per_epoch = int(test_num / cfg.BATCH_SIZE)

    train_ds = [train_data, train_steps_per_epoch]
    test_ds = [test_data, test_steps_per_epoch]

    # load target model
    tmodel = target_model()

    check_dir(cfg.GEN_SAVE_DIR)
    check_dir(cfg.DISC_SAVE_DIR)

    # function advgan
    # AdvGAN(train_data, test_data, train_steps_per_epoch, test_steps_per_epoch, tmodel)

    # class advgan, more efficiently
    GAN = AdvGan(train_ds, test_ds, tmodel)
    for epoch in range(cfg.EPOCHS):
        GAN.fit(epoch)
        GAN.test()

