import alphabets

trainroot = '/workspace/xqq/crnn_zh/data_generator/data_set/train_lmdb'
valroot = '/workspace/xqq/crnn_zh/data_generator/data_set/val_lmdb'
restore_ckpt = './expr/weights/crnn_Rec_done_82_15625.pth'

random_sample = True
keep_ratio = False
adam = False
adadelta = False
saveEpoch = 1
valInterval = 1000
n_test_disp = 10
displayInterval = 10
experiment = './expr/weights'
alphabet = alphabets.alphabet
crnn = ''
beta1 =0.5
lr = 0.0001
nEpochs = 300
nh = 256
imgW = 160
imgH = 32
batchSize = 64
workers = 8
