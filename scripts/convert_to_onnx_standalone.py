"""
独立版模型转 ONNX 脚本 -- 不依赖 escort 项目的模块结构。
可在 macOS ARM64 上运行。

使用方式:
    1. 创建临时虚拟环境:
       python3 -m venv /tmp/onnx_convert_env
       source /tmp/onnx_convert_env/bin/activate
       pip install torch torchvision numpy Pillow

    2. 运行转换:
       python scripts/convert_to_onnx_standalone.py

    3. Keras 模型单独转换 (可选, 需额外装 tensorflow + tf2onnx):
       pip install tensorflow tf2onnx
       python scripts/convert_to_onnx_standalone.py --keras

输出:
    scripts/output/word_detect.onnx
    scripts/output/word_compare.onnx
    scripts/output/face_direction.onnx  (需要 --keras 参数)

@author Sam
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
MODEL_DIR = os.path.join(ROOT_DIR, 'game_models', 'model')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- VGG16 backbone (内联定义, 与 escort/nets/vgg.py 一致) ----------

def _make_vgg_layers(cfg, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


_VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        return x


def _build_vgg16():
    return VGG(_make_vgg_layers(_VGG16_CFG))


# ---------- Siamese network (内联定义, 与 escort/nets/siamese.py 一致) ----------

def _get_output_length(input_length):
    filter_sizes = [2, 2, 2, 2, 2]
    stride = 2
    for i in range(5):
        input_length = (input_length + 0 - filter_sizes[i]) // stride + 1
    return input_length


class SiameseNet(nn.Module):
    def __init__(self, input_shape=(105, 105)):
        super().__init__()
        self.vgg = _build_vgg16()
        del self.vgg.avgpool
        del self.vgg.classifier
        flat_shape = 512 * _get_output_length(input_shape[1]) * _get_output_length(input_shape[0])
        self.fully_connect1 = nn.Linear(flat_shape, 512)
        self.fully_connect2 = nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x


class SiameseExportWrapper(nn.Module):
    """将 forward([x1, x2]) 包装为 forward(x1, x2)，并在输出端加 sigmoid。"""

    def __init__(self, net: SiameseNet):
        super().__init__()
        self.vgg_features = net.vgg.features
        self.fc1 = net.fully_connect1
        self.fc2 = net.fully_connect2

    def forward(self, x1, x2):
        x1 = self.vgg_features(x1)
        x2 = self.vgg_features(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


# ---------- 转换函数 ----------

def convert_yolov5():
    print('=' * 60)
    print('[1] 转换 YOLOv5 word.pt -> word_detect.onnx')
    print('=' * 60)

    model_path = os.path.join(MODEL_DIR, 'word.pt')
    yolov5_repo = os.path.join(ROOT_DIR, 'yolov5')
    output_path = os.path.join(OUTPUT_DIR, 'word_detect.onnx')

    if not os.path.exists(model_path):
        print(f'[SKIP] 模型文件不存在: {model_path}')
        return False

    if not os.path.exists(yolov5_repo):
        print(f'[SKIP] YOLOv5 仓库不存在: {yolov5_repo}')
        return False

    sys.path.insert(0, yolov5_repo)
    from models.experimental import attempt_load

    device = torch.device('cpu')
    model = attempt_load(model_path, device=device)
    model.eval()
    model.float()

    for m in model.model.modules():
        if hasattr(m, 'export'):
            m.export = True
        if hasattr(m, 'inplace'):
            m.inplace = False

    dummy = torch.zeros(1, 3, 640, 640)
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch'},
        },
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'[OK] word_detect.onnx ({size_mb:.1f} MB)')
    return True


def convert_siamese():
    print('=' * 60)
    print('[2] 转换 Siamese word_compare.pth -> word_compare.onnx')
    print('=' * 60)

    model_path = os.path.join(MODEL_DIR, 'word_compare.pth')
    output_path = os.path.join(OUTPUT_DIR, 'word_compare.onnx')

    if not os.path.exists(model_path):
        print(f'[SKIP] 模型文件不存在: {model_path}')
        return False

    device = torch.device('cpu')
    net = SiameseNet(input_shape=(105, 105))
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    wrapper = SiameseExportWrapper(net)
    wrapper.eval()

    x1 = torch.randn(1, 3, 105, 105)
    x2 = torch.randn(1, 3, 105, 105)

    torch.onnx.export(
        wrapper, (x1, x2), output_path,
        opset_version=12,
        input_names=['image_1', 'image_2'],
        output_names=['similarity'],
        dynamic_axes={
            'image_1': {0: 'batch'},
            'image_2': {0: 'batch'},
            'similarity': {0: 'batch'},
        },
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'[OK] word_compare.onnx ({size_mb:.1f} MB)')
    return True


def convert_keras():
    print('=' * 60)
    print('[3] 转换 Keras mhxy.h5 -> face_direction.onnx')
    print('=' * 60)

    model_path = os.path.join(MODEL_DIR, 'mhxy.h5')
    output_path = os.path.join(OUTPUT_DIR, 'face_direction.onnx')

    if not os.path.exists(model_path):
        print(f'[SKIP] 模型文件不存在: {model_path}')
        return False

    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError as e:
        print(f'[SKIP] 缺少依赖: {e}')
        print('  pip install tensorflow tf2onnx')
        return False

    model = tf.keras.models.load_model(model_path)
    spec = (tf.TensorSpec(model.input_shape, tf.float32, name='input'),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'[OK] face_direction.onnx ({size_mb:.1f} MB)')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型转 ONNX (独立版)')
    parser.add_argument('--keras', action='store_true', help='同时转换 Keras 模型 (需要 tensorflow + tf2onnx)')
    parser.add_argument('--only-keras', action='store_true', help='只转换 Keras 模型')
    args = parser.parse_args()

    print(f'模型目录: {MODEL_DIR}')
    print(f'输出目录: {OUTPUT_DIR}')
    print()

    results = {}
    if not args.only_keras:
        results['word_detect'] = convert_yolov5()
        print()
        results['word_compare'] = convert_siamese()
        print()

    if args.keras or args.only_keras:
        results['face_direction'] = convert_keras()
        print()

    print('=' * 60)
    print('结果汇总:')
    for name, ok in results.items():
        print(f'  {name}.onnx: {"OK" if ok else "SKIP/FAIL"}')

    if not args.keras and not args.only_keras:
        print()
        print('提示: Keras 模型未转换，如需转换请加 --keras 参数')
        print('  需要额外安装: pip install tensorflow tf2onnx')

    print()
    print('转换完成后，复制文件到 mhxy 项目:')
    print(f'  cp {OUTPUT_DIR}/*.onnx /Users/wujunxiong/pycharmProjects/mhxy/models/')
    print(f'  mkdir -p /Users/wujunxiong/pycharmProjects/mhxy/models/fonts')
    print(f'  cp {MODEL_DIR}/simsun.ttc {MODEL_DIR}/fsong.ttf /Users/wujunxiong/pycharmProjects/mhxy/models/fonts/')
