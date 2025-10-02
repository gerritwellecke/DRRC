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
    gitroot = str(conf.get_git_root()) + "/"

    for Key in ["Training", "Evaluation"]:
        print(f"Creating {Key} Data.")

        if Key == "Training":
            key = "training"
        else:
            key = "evaluation"

        for nr in range(conf["Data"]["Creation"]["datasets"]):
            # integrate system
            ks.generate_timeseries(
                t0=conf["Data"]["Creation"]["TemporalParameter"][key + "_transient"],
                tN=conf["Data"]["Creation"]["TemporalParameter"][key + "_length"]
                + conf["Data"]["Creation"]["TemporalParameter"][key + "_transient"],
            )

            # save data to file
            ks.save(
                gitroot
                + "/Data/"
                + f"{conf['Data']['model_dimension']}D_{conf['Data']['model_name']}/"
                + f"{Key}Data{nr}{conf['Data']['Usage']['fileformat']}"
            )

            # plot time series to check
            ks.plot_kymograph(
                f"Figures/1D_KuramotoSivashinsky/{Key}Data/{Key}Data{str(nr)}"
            )
            # ks.plot_conservation()


if __name__ == "__main__":
    main()
