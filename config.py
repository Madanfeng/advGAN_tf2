from easydict import EasyDict as edict

# class name list
CLASS_NAME = ['art_exposed', 'baby_naked', 'child_naked', 'exposed', 'feet', 'intimate_nannan',
              'intimate_nannv', 'intimate_nvnv', 'leggy', 'lutuitexie', 'midriff', 'mild_cleavage',
              'moderate_cleavage', 'naked', 'nan_midriff', 'nan_zhichuanneiyineiku', 'nanxiong',
              'normal', 'nv_zhichuanneiyineiku', 'porn', 'sex', 'sex_toy', 'sexy_qitaluolu',
              'sexy_tunbu', 'sexy_xiatitexie', 'sexy_xiongbutexie', 'shield_breast', 'sm']

__C                     = edict()
# Consumers can get config by: from config import cfg

cfg                     = __C

__C.TRAIN_ANNOT_PATH    = '/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/data/dataset/sex_new_10000.list'  # 通过 train_annot_path 标签文档来读取训练图片
__C.TEST_ANNOT_PATH     = '/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/data/dataset/sex_test_3000.list'
__C.SAVE_ADV_DIR        = './output/'     # 生成器生成对抗样本(包括扰动)保存的地址
# __C.LOG_PATH            = './log/sex/log_01.txt'  # 日志保存地址

__C.MODEL_PATH          = '/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/model/model.stage3.03-0.8788.v2_320_tf2.hdf5'
__C.GEN_SAVE_DIR        = "./weights/generator/"       # 在训练时，用于保存生成器模型的文件夹
__C.DISC_SAVE_DIR       = "./weights/discriminator/"   # 在训练时，用于保存判别器模型的文件夹
__C.GENERATOR_PATH      = "./weights/generator/last.h5"  # 在测试时，训练好的生成器模型地址


__C.GPU_IDS             = '6'

# Model parameters
__C.IMAGE_SIZE          = 320
__C.NUM_CHANNELS        = 3
__C.CLASS_NAME          = CLASS_NAME
__C.NUM_CLASS           = len(CLASS_NAME)


# Attack algorithm parameters
__C.BATCH_SIZE          = 32
__C.TARGET              = 17    # attack target label
__C.TARGETED            = True  # should we target one specific class? or just be wrong?
__C.CONFIDENCE          = 10    # how strong the adversarial example should be
__C.EPOCHS              = 10
__C.THRESH              = 0.5
__C.SMOOTH              = 0.0
