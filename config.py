from easydict import EasyDict as edict

# class name list
CLASS_NAME_28 = ['art_exposed', 'baby_naked', 'child_naked', 'exposed', 'feet', 'intimate_nannan',
                 'intimate_nannv', 'intimate_nvnv', 'leggy', 'lutuitexie', 'midriff', 'mild_cleavage',
                 'moderate_cleavage', 'naked', 'nan_midriff', 'nan_zhichuanneiyineiku', 'nanxiong',
                 'normal', 'nv_zhichuanneiyineiku', 'porn', 'sex', 'sex_toy', 'sexy_qitaluolu',
                 'sexy_tunbu', 'sexy_xiatitexie', 'sexy_xiongbutexie', 'shield_breast', 'sm']

__C                     = edict()
# Consumers can get config by: from config import cfg

cfg                     = __C

__C.TRAIN_ANNOT_PATH    = '/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/data/dataset/sex_new_10000.list'  # 通过 train_annot_path 标签文档来读取训练图片
__C.TEST_ANNOT_PATH     = '/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/data/dataset/sex_test_3000.list'
__C.SAVE_ADV_DIR        = './output/con20_2_gen4_class/12/'     # 生成器生成的对抗样本(包括扰动)保存的地址

__C.MODEL_PATH          = '/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/model/model.stage3.03-0.8788.v2_320_tf2.hdf5'
__C.GEN_SAVE_DIR        = "./weights/con20_2_gen4_class_moreadvloss/generator/"       # 在训练时，用于保存生成器模型的文件夹
__C.DISC_SAVE_DIR       = "./weights/con20_2_gen4_class_moreadvloss/discriminator/"   # 在训练时，用于保存判别器模型的文件夹
__C.GENERATOR_PATH      = "./weights/con20_2_gen4_class/generator/12.h5"  # 在测试时，训练好的生成器模型地址


__C.GPU_IDS             = '4'

# Model parameters
__C.BATCH_SIZE          = 32
__C.IMAGE_SIZE          = 320
__C.NUM_CHANNELS        = 3
__C.CLASS_NAME          = CLASS_NAME_28
__C.NUM_CLASS           = len(CLASS_NAME_28)


# Attack algorithm parameters
__C.TARGET              = 0     # attack target label
__C.TARGETED            = True  # should we target one specific class? or just be wrong?
__C.CONFIDENCE          = 20    # how strong the adversarial example should be
__C.EPOCHS              = 50
__C.THRESH              = 0.5
__C.SMOOTH              = 0.0
__C.ALPHA               = 1.0   # control g_loss_fake, the bigger, the stronger the generator
__C.BETA                = 5.0   # control l_perturb, the bigger, the adversarial example closer to the original image

