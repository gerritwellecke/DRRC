import argparse
import logging
import os
import pathlib
import pickle

# import gc
import sys
import time
from datetime import timedelta

# import time
from pathlib import Path

# import matplotlib.pyplot as plt
# import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.pyplot import cm
from numpy.random import SeedSequence

import drrc.DR_Trafo_extension
import drrc.model
from drrc.config import Config

# from scipy.signal import find_peaks


valwarm = None
learnwarm = None


def load_data(conf, taskid, repeatid, train, dtype):
    global valwarm
    global learnwarm
    confdata = conf["Training"] if train else conf["Evaluation"]
    scanlist = conf.param_scan_list(task_id=taskid)
    path = pathlib.Path(
        conf.get_git_root(),
        confdata["FilePath"] + str(repeatid) + confdata["FileFormat"],
    )
    dataall = np.load(path)
    try:
        vals = dataall["vars"]
    except:
        vals = dataall

    end = int(confdata["Length"] / confdata["dt"])
    if len(vals.shape) == 2:
        data = np.array(vals[:end], dtype=dtype)
    else:
        data = np.array(vals[:end, 0, ...], dtype=dtype)
    if train:
        learnwarm = int(confdata["Transient"] / confdata["dt"])
        logging.info("train")
    else:
        valwarm = int(confdata["Transient"] / confdata["dt"])
        logging.info("evaluation")
    logging.info("datashape: " + str(data.shape))
    return data


def writescores(filename, seeds, valtimes):
    # write numpy arrays to file
    # as such: seed \t valtime1 \t valtime2 \t valtime3 ...
    with open(
        filename,
        "w",
    ) as f:
        titlestr = "seeds\tvaltimes...\n"
        f.write(titlestr)
        # write seed and then all evaluation results line by line
        for i, seed in enumerate(seeds):
            linewrite = f"{seed}"
            for eval in valtimes[i]:
                linewrite += f"\t{eval}"
            linewrite += "\n"  # trailing newline is not a problem for np.loadtxt
            f.write(linewrite)


def main(args, conf, taskid):
    try:
        maxL = float(args.data_file[:-4].split("_")[-1])
    except:
        maxL = 300

    winsize = args.winsize
    model = None

    if np.prod(args.replicas) == 1:
        winsize = 0

    seeds = []
    valtimes = np.full(
        (conf["Training"]["Datasets"], conf["Evaluation"]["Datasets"]),
        np.nan,
        dtype=np.float32,
    )
    # execute same hyperparameter multiple times for statistics
    for trainid in range(conf["Training"]["Datasets"]):
        # load in train and evaluation data
        data_train = load_data(conf, taskid, trainid, True, args.dtype)
        origshape = data_train.shape[1:]

        if len(args.replicas) == 1:
            args.replicas = (args.replicas[0],) + tuple(
                1 for i in range(len(origshape) - 1)
            )
        else:
            assert len(args.replicas) == len(origshape)

        modelstr = f"Model_{args.taskid}_{trainid}_{args.nodes}_{args.spectral}_{args.degree}_{args.inscale}_{args.fracin}_{args.replicas}_{args.winsize}_{args.leakage}_{args.regularization}_{args.combine}.pkl"
        try:
            os.makedirs(Path(conf["Saving"]["OutputDirectory"]))
        except:
            pass
        savefilename = Path(
            Path(conf["Saving"]["OutputDirectory"]), modelstr
        )  # filename to safe model under (currently disabled on cluster further down)

        # initialize Model
        logging.info("Modelname: " + modelstr)
        if os.path.isfile(savefilename) and not args.overwrite:
            logging.info("Loading Model from File!")
            model = pickle.load(open(savefilename, "rb"))
        else:
            logging.info("Training Model!")

            if args.transform == "fft":
                logging.info("fft used")
                modestr = "fft"
            elif args.transform == "pca":
                logging.info("pca used")
                modestr = "pca"
            elif args.transform == "identity":
                logging.info("no transformation used")
                modestr = "identity"
            else:
                raise Exception("transformation: " + args.transform + " not recognized")
            transforms = drrc.DR_Trafo_extension.return_transforms(modestr)

            assert np.all(np.mod(origshape, args.replicas) == 0)
            seedseq = SeedSequence()
            try:
                model = drrc.model.MultiModel(
                    input_size=origshape,
                    reservoir_grid=args.replicas,
                    window_size=winsize,
                    nodes=args.nodes,
                    transform_functions=transforms,
                    fraction_of_dimension=args.fracin,
                    spectral_radius=args.spectral,
                    degree=args.degree,
                    input_scaling=args.inscale,
                    leakage=args.leakage,
                    regularization=args.regularization,
                    bias_scaling=args.biasscale,
                    combine=args.combine,
                    input_adj_same=(not args.inputadjdiff),
                    signal_fit=args.fitsignal,
                    sparse=(not args.dense),
                    dtype=args.dtype,
                    entropy=seedseq,
                )
            except Exception as ex:
                logging.error(
                    "error initializing reservoir {} with seed {}\n".format(
                        trainid, seedseq.entropy
                    )
                )
                logging.error(ex)
                # continue here with the next reservoir if any error occured
                seeds += [seedseq.entropy]
                continue

        seeds += [model.get_entropy()]

        # fill in ghost cells to training data
        data_train = drrc.DR_Trafo_extension.IncludeBoundary(data_train, winsize)

        # train if model was not trained before
        if not model.trained_before:
            if not args.trainonly:
                savefilename = None  # disable model saving
            beforetime = time.time()
            model.train(data=data_train, transient=learnwarm, savefilename=savefilename)
            endtime = time.time()
            logging.info("took time to train: " + str(endtime - beforetime))

        if args.trainonly:
            continue

        # assert atleast one evaluation run otherwise the below writing of metrics wont work
        assert conf["Evaluation"]["Datasets"] > 0

        # evaluate over multiple evaluation datasets to gather metrics
        beforetime = time.time()
        for evalid in range(conf["Evaluation"]["Datasets"]):
            data_val = load_data(conf, taskid, evalid, False, args.dtype)
            data_val = drrc.DR_Trafo_extension.IncludeBoundary(data_val, winsize)

            # predict over validation data
            stopval = conf["Evaluation"]["ErrorStop"]

            def analyze(t, predict):
                difference = predict - data_val[t + 1]
                # MeanVariation measures: np.linalg.norm(np.std(data_val, axis=0))
                # as an average over the entire evaluation set
                meandiff = (
                    np.sqrt(np.sum(difference**2))
                    / conf["Evaluation"]["MeanVariation"]
                )
                if meandiff > stopval:
                    return False
                return True

            valtimenow, predictions = model.predict_timeseries(
                data_val[:valwarm], data_val.shape[0], safe=False, analyzer=analyze
            )
            valtimes[trainid, evalid] = valtimenow * conf["Evaluation"]["dt"]

        endtime = time.time()
        logging.info("took time to evaluate: " + str(endtime - beforetime))

        scorefilename = str(
            Path(
                conf["Saving"]["OutputDirectory"],
                "score_{}.txt".format(args.taskid),
            )
        )
    writescores(scorefilename, seeds, valtimes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # params for "Data/2D_AlievPanfilov/AlievPanfilov_StableSpiral.npz":
    # Model_9000_0.5_2_0.05_4_probe60_4_data.npy

    # Model_9000_0.1_2_0.01_1_probe60_1_data
    # Model_9000_0.01_2_0.05_2_probe60_2_data.npy
    # Model_9000_0.5_2_0.05_4_probe60_4_data.npy
    # datadef = "Data/1D_KuramotoSivashinsky/KS_t_60.npy"
    # datadef = "Data/2D_AlievPanfilov/AlievPanfilov_StableSpiral.npz"
    # datadef = "Data/2D_AlievPanfilov/AlievPanfilov_Chaos.npz"

    # np.random.seed(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=str, default=None)
    parser.add_argument("taskid", type=int, default=None)
    args = parser.parse_args()
    # args.suffix = "_" + args.suffix

    # load configuration
    conf = Config(Path(args.yaml).absolute())
    logging.info(f"Task Id: {args.taskid}")
    if int(args.taskid) == 0:
        # for quick debugging
        logging.info(
            f"Array job with {conf.param_scan_len()} executions.\n\nCONFIGURATION:\n"
        )
        logging.info(str(conf))

        # ensure that datadir exists
        conf.make_jobscript_datadir()
        # generate the cluster submission script
        jobscript_path = conf.generate_submission_script_from_YAML()
        print(jobscript_path)
    else:
        logging.info(
            "dict1: "
            + str(conf["Reservoir"])
            + " scans: "
            + str(conf.param_scan_list(task_id=args.taskid))
        )
        dictjoin = {
            **conf["Reservoir"],
            **conf.param_scan_list(task_id=args.taskid),
        }
        for key in dictjoin.keys():
            setattr(args, key, dictjoin[key])

        program_start = time.time()
        main(args, conf, taskid=int(args.taskid))
        program_end = time.time()
        logging.info(
            "took time for everything: "
            + str(timedelta(seconds=program_end - program_start))
        )
