import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from DFN import DFN
import time

parser = argparse.ArgumentParser(description="DFN_Test")
parser.add_argument("--logdir", type=str, default="..", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="..", help='path to training data')
parser.add_argument("--save_path", type=str, default="..", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=7, help='number of recursive stages')
parser.add_argument("--number_blocks", type=int, default=2, help='number of feedback blocks')
parser.add_argument("--self_ensemble", type=bool, default=False, help='use self_ensemble or not')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def _transform(v, op):
    v = v.float()
    v2np = v.data.cpu().numpy()
    if op == 'v':
        tfnp = v2np[:, :, :, ::-1].copy()
    elif op == 'h':
        tfnp = v2np[:, :, ::-1, :].copy()

    ret = torch.Tensor(tfnp)
    ret = ret.cuda()

    return ret




def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = DFN(opt.recurrent_iter, opt.number_blocks)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))

    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad(): #
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                if opt.self_ensemble:
                    lr_list = [y]
                    for tf in 'v', 'h':
                        lr_list.extend([_transform(t, tf) for t in lr_list])

                    sr_list = []
                    for aug in lr_list:
                        #sr, _ = model(aug)
                        sr, slist=model(aug)
                        sr_list.append(sr)

                    for i in range(len(sr_list)):
                        if i % 4 > 1:
                            sr_list[i] = _transform(sr_list[i], 'h')
                        if (i % 4) % 2 == 1:
                            sr_list[i] = _transform(sr_list[i], 'v')

                    # output_cat = torch.cat(sr_list, dim=0)
                    # output = output_cat.mean(dim=0, keepdim=True)
                    out = sum(sr_list) / len(sr_list)
                else:
                    out, _ = model(y)

                #out, _ = model(y)

                out = torch.clamp(out, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    main()

