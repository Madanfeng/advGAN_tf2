import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from collections import Iterable

from config import cfg
from data import Dataset
from models import Generator, Discriminator, target_model

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS

strategy = tf.distribute.MirroredStrategy()


def check_dir(dir):
    if not os.path.exists(dir):
        print("Make", dir)
        os.makedirs(dir)


with strategy.scope():
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

    loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=cfg.BATCH_SIZE)


# train advGAN
def AdvGAN(train_data, test_data, tmodel):
    thresh = cfg.THRESH
    smooth = cfg.SMOOTH
    is_targeted = cfg.TARGETED

    with strategy.scope():
        discriminator = Discriminator()
        generator = Generator()

        # define optimizers
        d_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)
        g_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

        def train_step(images, targets_tensor_onehot):
            # LOSS DEFINITIONS
            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                disc_tape.watch(discriminator.trainable_variables)
                gen_tape.watch(generator.trainable_variables)

                # generate perturbation, add to original input image(s)
                perturb = tf.clip_by_value(generator(images, training=True), -thresh, thresh)
                images_perturbed = perturb + images
                images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)

                # pass real and perturbed image to discriminator and the target model
                d_real_logits = discriminator(images, training=True)
                d_real_probs = tf.nn.sigmoid(d_real_logits)

                d_fake_logits = discriminator(images_perturbed, training=True)
                d_fake_probs = tf.nn.sigmoid(d_fake_logits)

                # pass real and perturbed images to the model we are trying to fool
                # f_real_logits, f_real_probs = tmodel.predict_logits(images)
                f_fake_logits, f_fake_probs = tmodel.predict_logits(images_perturbed)

                # generate labels for discriminator (optionally smooth labels for stability)
                d_labels_real = tf.ones_like(d_real_probs) * (1 - smooth)
                d_labels_fake = tf.zeros_like(d_fake_probs)

                # discriminator loss
                d_loss_real = compute_loss(d_labels_real, d_real_probs)
                d_loss_fake = compute_loss(d_labels_fake, d_fake_probs)
                # d_loss_real = tf.keras.losses.MSE(y_pred=d_real_probs, y_true=d_labels_real)
                # d_loss_fake = tf.keras.losses.MSE(y_pred=d_fake_probs, y_true=d_labels_fake)
                d_loss = d_loss_real + d_loss_fake

                # generator loss
                g_loss_fake = compute_loss(tf.ones_like(d_fake_probs), d_fake_probs)
                # g_loss_fake = tf.losses.MSE(y_pred=d_fake_probs, y_true=tf.ones_like(d_fake_probs))

                # perturbation loss (minimize overall perturbation)
                l_perturb = perturb_loss(perturb, thresh)

                # adversarial loss (encourage misclassification)
                l_adv = adv_loss(f_fake_logits, targets_tensor_onehot, is_targeted)

                # weights for generator loss function
                alpha = 1.0
                beta = 5.0
                g_loss = l_adv + alpha * g_loss_fake + beta * l_perturb

            # train the discriminator
            disc_grad = disc_tape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

            # train the generator
            gen_grad = gen_tape.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))

            return d_loss, g_loss

        def test_step(images):
            perturbs = tf.clip_by_value(generator(images, training=False), -thresh, thresh)
            images_perturbed = perturbs + images
            images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)

            probs = tmodel.predict_softmax(images_perturbed)

            return probs, perturbs

        def distributed_train_step(images, targets_tensor_onehot):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(images, targets_tensor_onehot))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        def distributed_test_step(images):
            return strategy.experimental_run_v2(test_step, args=(images,))

        print("Train GAN...")
        for epoch in range(cfg.EPOCHS):

            if epoch == int(cfg.EPOCHS * 0.7):
                d_opt = tf.keras.optimizers.Adam(4e-5, beta_1=0.5)
                g_opt = tf.keras.optimizers.Adam(4e-5, beta_1=0.5)

            pbar = tqdm(train_data)
            for images, labels, targets, paths in pbar:
                # convert to one-hot
                targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
                targets_tensor_onehot = tf.one_hot(targets_tensor, depth=train_data.class_num)

                d_loss, g_loss = distributed_train_step(images, targets_tensor_onehot)

                pbar.set_description("epoch: %d, gen_loss: %.8f, d_loss: %.8f" % (epoch, g_loss.numpy(), d_loss.numpy()))

            # test
            suc_num = 0
            sum_num = 0
            p_dis = 0
            sum_prob = 0

            for images, labels, targets, paths in test_data:

                probs, perturbs = distributed_test_step(images)

                for i, prob in enumerate(probs):
                    sum_prob += prob.numpy()[np.int(targets[i])]
                    sum_num += 1
                    ind = tf.argmax(prob)
                    if ind == targets[i]: suc_num += 1

                    p_dis += np.sum((perturbs[i] ** 2) ** .5)

            acc = suc_num / sum_num
            dis = p_dis / sum_num
            ave_prob = sum_prob / sum_num
            print("Attack Success Rate:", acc, "Average Confidence:", ave_prob, "Perturb Distance:", dis)

            # print("Save last model...")
            # discriminator.save(cfg.DISC_SAVE_DIR + "lsat.h5", save_format='h5')
            # generator.save(cfg.GEN_SAVE_DIR + "last.h5", save_format='h5')

            if epoch % 1 == 0:
                print("Save model...")
                discriminator.save(cfg.DISC_SAVE_DIR+str(epoch)+".h5", save_format='h5')
                generator.save(cfg.GEN_SAVE_DIR+str(epoch)+".h5", save_format='h5')


if __name__ == '__main__':
    # load dataset
    train_data = Dataset(istrain=True)
    test_data = Dataset(istrain=False)
    print(isinstance(train_data, Iterable))

    train_dist_dataset = strategy.experimental_distribute_dataset(train_data)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_data)

    # load target model
    with strategy.scope():
        tmodel = target_model()

    check_dir(cfg.GEN_SAVE_DIR)
    check_dir(cfg.DISC_SAVE_DIR)
    AdvGAN(train_dist_dataset, test_dist_dataset, tmodel)

