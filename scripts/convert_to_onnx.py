"""
将 mhxy-escort 的 3 个模型转换为 ONNX 格式，供 mhxy 项目使用。
在 mhxy-escort 的 Python 环境下运行（需要 PyTorch + TensorFlow）。

使用方式:
    cd /Users/wujunxiong/pycharmProjects/mhxy-escort
    python scripts/convert_to_onnx.py

输出:
    scripts/output/word_detect.onnx
    scripts/output/word_compare.onnx
    scripts/output/face_direction.onnx

@author Sam
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn

from game_models.source import get_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_yolov5_word_detect():
    """word.pt (YOLOv5) → word_detect.onnx"""
    print('=' * 60)
    print('[1/3] 转换 YOLOv5 word.pt → word_detect.onnx')
    print('=' * 60)

    yolov5_repo = os.path.join(ROOT_DIR, 'yolov5')
    model_path = get_model('word')
    output_path = os.path.join(OUTPUT_DIR, 'word_detect.onnx')

    model = torch.hub.load(yolov5_repo, 'custom', path=model_path, source='local')
    model.eval()
    model.model.float()

    img_size = 640
    dummy_input = torch.zeros(1, 3, img_size, img_size)

    sys.path.insert(0, yolov5_repo)
    from models.experimental import attempt_load
    pt_model = attempt_load(model_path, device=torch.device('cpu'))
    pt_model.eval()
    pt_model.float()

    for m in pt_model.model.modules():
        if hasattr(m, 'export'):
            m.export = True
        if hasattr(m, 'inplace'):
            m.inplace = False

    torch.onnx.export(
        pt_model,
        dummy_input,
        output_path,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch'},
        },
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'[OK] word_detect.onnx ({file_size:.1f} MB) -> {output_path}')


class SiameseOnnxWrapper(nn.Module):
    """将 Siamese 的 forward([x1, x2]) 包装为 forward(x1, x2)，适配 ONNX 导出。"""

    def __init__(self, siamese_model):
        super().__init__()
        self.vgg = siamese_model.vgg
        self.fully_connect1 = siamese_model.fully_connect1
        self.fully_connect2 = siamese_model.fully_connect2

    def forward(self, x1, x2):
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return torch.sigmoid(x)


def convert_siamese():
    """word_compare.pth (Siamese VGG16) → word_compare.onnx"""
    print('=' * 60)
    print('[2/3] 转换 Siamese word_compare.pth → word_compare.onnx')
    print('=' * 60)

    from nets.siamese import Siamese as SiameseNet

    input_shape = [105, 105]
    model_path = get_model('siamese')
    output_path = os.path.join(OUTPUT_DIR, 'word_compare.onnx')

    device = torch.device('cpu')
    model = SiameseNet(input_shape)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    wrapper = SiameseOnnxWrapper(model)
    wrapper.eval()

    dummy_x1 = torch.randn(1, 3, input_shape[0], input_shape[1])
    dummy_x2 = torch.randn(1, 3, input_shape[0], input_shape[1])

    torch.onnx.export(
        wrapper,
        (dummy_x1, dummy_x2),
        output_path,
        opset_version=12,
        input_names=['image_1', 'image_2'],
        output_names=['similarity'],
        dynamic_axes={
            'image_1': {0: 'batch'},
            'image_2': {0: 'batch'},
            'similarity': {0: 'batch'},
        },
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'[OK] word_compare.onnx ({file_size:.1f} MB) -> {output_path}')


def convert_keras_face():
    """mhxy.h5 (Keras CNN) → face_direction.onnx"""
    print('=' * 60)
    print('[3/3] 转换 Keras mhxy.h5 → face_direction.onnx')
    print('=' * 60)

    output_path = os.path.join(OUTPUT_DIR, 'face_direction.onnx')
    model_path = get_model('four_people')

    try:
        import tf2onnx
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        model.summary()

        spec = (tf.TensorSpec(model.input_shape, tf.float32, name='input'),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f'[OK] face_direction.onnx ({file_size:.1f} MB) -> {output_path}')

    except ImportError:
        print('[WARN] tf2onnx 未安装，尝试命令行方式...')
        print(f'  pip install tf2onnx')
        print(f'  python -m tf2onnx.convert --keras {model_path} --output {output_path}')
        return False

    return True


if __name__ == '__main__':
    print(f'ROOT_DIR: {ROOT_DIR}')
    print(f'OUTPUT_DIR: {OUTPUT_DIR}')
    print()

    convert_yolov5_word_detect()
    print()
    convert_siamese()
    print()
    convert_keras_face()

    print()
    print('=' * 60)
    print('转换完成！请将以下文件复制到 mhxy/models/ 目录：')
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.onnx'):
            full_path = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(full_path) / (1024 * 1024)
            print(f'  {f} ({size:.1f} MB)')
    print()
    print('同时需要复制字体文件到 mhxy/models/fonts/：')
    print(f'  {get_model("simsun")}')
    print(f'  {get_model("fsong")}')
