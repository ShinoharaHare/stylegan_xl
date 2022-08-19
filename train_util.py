import os
from pathlib import Path
from types import SimpleNamespace

import click


class Args(SimpleNamespace):
    @property
    def batch_gpu(self):
        return self.__dict__['batch-gpu']

    @batch_gpu.setter
    def batch_gpu(self, value):
        self.__dict__['batch-gpu'] = value

    @property
    def snap_img(self):
        return self.__dict__['snap-img']

    @snap_img.setter
    def snap_img(self, value):
        self.__dict__['snap-img'] = value

    @property
    def metric_ticks(self):
        return self.__dict__['metric-ticks']

    @metric_ticks.setter
    def metric_ticks(self, value):
        self.__dict__['metric-ticks'] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def copy(self):
        return Args(**self.__dict__)

    def build_command(self, background: bool = True):
        cmd = ''
        cmd += 'nohup' if background else ''
        cmd += ' python train.py'
        for k, v in self.__dict__.items():
            if v is None:
                continue 
            elif type(v) == bool:
                cmd += f' --{k}' if v else ''
            else:
                cmd += f' --{k}={v}'
        cmd += ' > /dev/null 2>&1 &' if background else ''
        return cmd

@click.command()
@click.argument('background', type=bool, default=False)
def main(background=False):
    cwd = os.getcwd()
    path = Path(__file__).absolute()
    os.chdir(path.parent)
    # os.environ['TORCH_HOME'] = '/workspace/.cache/torch'
    # os.environ['TORCH_EXTENSIONS_DIR'] = '/workspace/.cache/torch_extensions'

    args_base = Args()
    args_base.cfg = 'stylegan3-t'
    args_base.outdir = os.path.join(cwd, 'training/stylegan_xl')
    args_base.workers = 2
    args_base.gpus = 1
    args_base.batch_gpu = 4
    args_base.snap = 1
    args_base.snap_img = 50
    args_base.metric_ticks = 50
    args_base.metrics = 'fid50k_full' # none fid50k_full

    # 16x16
    args_16x16 = args_base.copy()
    args_16x16.data = os.path.join(cwd, 'data/v2/top-cropped/16x16.zip')
    args_16x16.batch = 2048
    args_16x16.stem = True

    # 32x32
    args_32x32 = args_base.copy()
    args_32x32.data = os.path.join(cwd, 'data/v2/top-cropped/32x32.zip')
    args_32x32.batch = 2048
    args_32x32.superres = True
    args_32x32.up_factor = 2
    args_32x32.path_stem = os.path.join(cwd, 'training/stylegan_xl/00000-stylegan3-t-16x16-gpus1-batch2048/best_model.pkl')

    # 64x64
    args_64x64 = args_base.copy()
    args_64x64.data = os.path.join(cwd, 'data/v2/top-cropped/64x64.zip')
    args_64x64.batch = 256
    args_64x64.superres = True
    args_64x64.up_factor = 2
    args_64x64.path_stem = os.path.join(cwd, 'training/stylegan_xl/00001-stylegan3-t-32x32-gpus1-batch2048/best_model.pkl')

    # 128x128
    args_128x128 = args_base.copy()
    args_128x128.data = os.path.join(cwd, 'data/v2/top-cropped/128x128.zip')
    args_128x128.batch = 256
    args_128x128.superres = True
    args_128x128.up_factor = 2
    args_128x128.path_stem = os.path.join(cwd, 'training/stylegan_xl/00002-stylegan3-t-64x64-gpus1-batch256/best_model.pkl')

    # 256x256
    args_256x256 = args_base.copy()
    args_256x256.data = os.path.join(cwd, 'data/v2/top-cropped/256x256.zip')
    args_256x256.batch = 256
    args_256x256.superres = True
    args_256x256.up_factor = 2
    args_256x256.path_stem = os.path.join(cwd, 'training/stylegan_xl/00002-stylegan3-t-128x128-gpus1-batch256/best_model.pkl')

    # 512x512
    args_512x512 = args_base.copy()
    args_512x512.data = os.path.join(cwd, 'data/v2/top-cropped/512x512.zip')
    args_512x512.batch = 128
    args_512x512.superres = True
    args_512x512.up_factor = 2
    args_512x512.path_stem = os.path.join(cwd, 'training/stylegan_xl/00003-stylegan3-t-256x256-gpus1-batch256/best_model.pkl')

    # 1024x1024
    args_1024x1024 = args_base.copy()
    args_1024x1024.data = os.path.join(cwd, 'data/v2/top-cropped/1024x1024.zip')
    args_1024x1024.batch = 128
    args_1024x1024.superres = True
    args_1024x1024.up_factor = 2
    args_1024x1024.path_stem = os.path.join(cwd, 'training/stylegan_xl/00004-stylegan3-t-512x512-gpus1-batch128/best_model.pkl')
    
    # Run
    args = args_16x16
    os.system(args.build_command(background))

if __name__ == '__main__':
    main()