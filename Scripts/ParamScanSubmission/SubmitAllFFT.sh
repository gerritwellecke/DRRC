#!/bin/bash

configs=`ls ../ParamScanYml/KS_SmNod_HypOptimization_PCA*.yml`

for c in $configs; do
    ./ReservoirOnCluster.sh $c
done
