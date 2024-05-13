import os.path as osp
import time
from typing import Optional

from tensorboardX import SummaryWriter


def init_tensorboard(logdir="runs", name: Optional[str] = None):
    if name is not None:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        return SummaryWriter(osp.join(logdir, f"{time_str}_{name}"))
    else:
        import socket
        from datetime import datetime

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        return SummaryWriter(
            osp.join(logdir, current_time + "_" + socket.gethostname())
        )
