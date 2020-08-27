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
__C.MODEL_PATH          = '/data/project/madanfeng/test/nn_robust_attacks/attack_tf2/model/model.stage3.03-0.8788.v2_320_tf2.hdf5'
# __C.SAVE_ADV_PATH       = './output/sex/sex_01/'     # 对抗样本保存的地址
# __C.LOG_PATH            = './log/sex/log_01.txt'  # 日志保存地址
__C.GEN_SAVE_DIR        = "./weights/generator/"
__C.DISC_SAVE_DIR       = "./weights/discriminator/"


__C.GPU_IDS             = '6'

# Model parameters
__C.BATCH_SIZE          = 32
__C.IMAGE_SIZE          = 320
__C.NUM_CHANNELS        = 3
__C.CLASS_NAME          = CLASS_NAME_28
__C.NUM_CLASS           = len(CLASS_NAME_28)


# Attack algorithm parameters
__C.TARGET              = 17    # attack target label
__C.TARGETED            = True  # should we target one specific class? or just be wrong?
__C.CONFIDENCE          = 10    # how strong the adversarial example should be
__C.EPOCHS              = 10
__C.THRESH              = 0.5
__C.SMOOTH              = 0.0
