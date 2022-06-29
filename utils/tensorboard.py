from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def get_summary_writer_for(model_name, base_dir="/home/users/sadler/cache/tensorboard-logdir/"):
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d/%H%M")
    print("now =", dt_string)
    log_dir = base_dir + model_name + "/" + dt_string
    return SummaryWriter(log_dir)
