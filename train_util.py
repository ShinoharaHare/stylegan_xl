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
    os.environ['TORCH_HOME'] = '/workspace/.cache/torch'
    os.environ['TORCH_EXTENSIONS_DIR'] = '/workspace/.cache/torch_extensions'

    args = Args()
    args.cfg = 'stylegan3-t'
    args.outdir = os.path.join(cwd, 'training')
    args.data = os.path.join(cwd, 'data/16x16.zip')
    args.workers = 4
    args.gpus = 4
    args.batch = 2048
    args.batch_gpu = 4
    args.snap = 1
    args.snap_img = None
    args.metric_ticks = 100
    args.kimg = 10000
    args.stem = True
    args.metrics = 'fid50k_full' # none fid50k_full

    os.system(args.build_command(background))

if __name__ == '__main__':
    main()