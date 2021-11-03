# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""predict resnet."""
from src.resnet import resnet101 as resnet
from src.config import config3 as config
from src import class_dir

import cv2
import argparse
import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net



parser = argparse.ArgumentParser(description='Image classification')

parser.add_argument('--checkpoint_path', required=False, type=str, default='./resnet-50_7184.ckpt', help='Checkpoint file path')
parser.add_argument('--image_path', required=False, type=str, default='./屏幕快照 2020-02-14 上午11.37.47.png', help='Image path')
parser.add_argument('--device_target', type=str, default='GPU', help='Device target. Default: GPU')
args_opt = parser.parse_args()

set_seed(1)


def _crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def _normalize(img, mean, std):
    # This method is borrowed from:
    #   https://github.com/open-mmlab/mmcv/blob/master/mmcv/image/photometric.py
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def data_preprocess(img_path):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (256, 256))
    img = _crop_center(img, 224, 224)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
    img = img.transpose(2, 0, 1)

    return img


if __name__ == '__main__':
    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # preprocess the image
    img = data_preprocess(args_opt.image_path)
    # predict model
    res = net(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()

    predict_label = class_dir.label_list[res[0].argmax()]
    print("预测的商品标签为:\n\t"+predict_label+"\n")
    # print("预测的商品标签为:\n\t"+str(res[0].argmax())+"\n")
    # print("预测的商品标签为:\n\t"+str(res[0].argmax())+"\n")
