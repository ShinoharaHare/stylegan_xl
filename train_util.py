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
    def snap_image(self):
        return self.__dict__['snap-image']

    @snap_image.setter
    def snap_image(self, value):
        self.__dict__['snap-image'] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def build_command(self, background: bool = True):
        cmd = ''
        cmd += 'nohup' if background else ''
        cmd += ' python train.py'
        for k, v in self.__dict__.items():
            if type(v) == bool:
                cmd += f' --{k}' if v else ''
            else:
                cmd += f' --{k}={v}'
        cmd += ' > /dev/null 2>&1 &' if background else ''
        return cmd

@click.command()
@click.argument('background', type=bool, default=False)
def main(background=False):
    path = Path(__file__).absolute()
    os.chdir(path.parent)
    os.environ['TORCH_EXTENSIONS_DIR'] = str(path.parent.joinpath('.cache'))

    args = Args()
    args.cfg = 'stylegan3-t'
    args.outdir = '../training/16x16'
    args.data = '../data/v2/16x16.zip'
    args.workers = 2
    args.gpus = 1
    args.batch = 2048
    args.batch_gpu = 8
    args.snap = 1
    args.snap_image = 10
    args.kimg = 10000
    args.stem = True
    args.metrics = 'none' # none fid50k_full

    os.system(args.build_command(background))

if __name__ == '__main__':
    main()