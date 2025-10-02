import argparse
import os
from pathlib import Path

from drrc.config import Config
from drrc.kuramoto_sivashinsky import KuramotoSivashinsky


def main():
    # use argparse for the absolute path of the config
    parser = argparse.ArgumentParser(description="Produce time series from config file")
    parser.add_argument("filepath", help="Path and Name to config file", type=str)
    args = parser.parse_args()
    path = Path(args.filepath)

    task_id = os.environ.get("SGE_TASK_ID")
    if task_id is None:
        task_id = 0
    else:
        task_id = int(task_id)

    # create configuration & system instances
    conf = Config(path)
    ks = KuramotoSivashinsky(
        config=conf,
        method=conf["Data"]["Creation"]["Method"],
        **conf["Data"]["Creation"]["SystemParameters"],
        task_id=task_id,
        cluster_save=False,
    )

    # integrate system
    try:
        length = conf["Data"]["Creation"]["TemporalParameter"]["length"]
        transient = conf["Data"]["Creation"]["TemporalParameter"]["transient"]
    except KeyError:
        length = conf["Data"]["Creation"]["TemporalParameter"]["training_length"]
        transient = conf["Data"]["Creation"]["TemporalParameter"]["training_transient"]

    ks.generate_timeseries(t0=transient, tN=length + transient)

    # save data to file
    ks.save(
        str(conf.get_git_root())
        + "/"
        + "/Data/"
        + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
        + f"Data{conf['Data']['Usage']['fileformat']}"
    )

    # plot time series to check
    ks.plot_kymograph()
    # ks.plot_conservation()


if __name__ == "__main__":
    main()
