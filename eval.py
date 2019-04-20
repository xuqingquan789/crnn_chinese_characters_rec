import torch
import numpy as np
import torch.backends.cudnn as cudnn
import dataset
import params
import utils
import random
from models import crnn
from torch.autograd import Variable


def val(net, dataset):
    print('Start eval...')
    for p in crnn.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=64, num_workers=int(params.workers))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    count = 0

    for i in range(len(data_loader)):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        list_1 = []
        for i in cpu_texts:
            list_1.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1
            else:
                print('{} pred:{} ==> label:{}'.format(count, pred, target))
            count += 1
    
    # raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    # for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
    #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / len(data_loader.dataset)
    print('Num correct: %d, accuray: %f' % (n_correct, accuracy))


manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = True
 
test_dataset = dataset.lmdbDataset(
        root=params.valroot, transform=dataset.resizeNormalize((160, 32)))

converter = utils.strLabelConverter(params.alphabet)
nclass = len(params.alphabet) + 1
nc = 1
crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh).cuda()
text = torch.IntTensor(params.batchSize * 5)
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH).cuda()
length = torch.IntTensor(params.batchSize)

crnn.load_state_dict(torch.load(params.restore_ckpt))

val(crnn, test_dataset)
