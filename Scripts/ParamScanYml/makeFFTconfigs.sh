#!/bin/bash

files=`ls KS_SmNod_HypOptimization_PCA*.yml`

# copy files
for f in $files; do
    new_f=`echo $f|sed -e 's/\(.*\)SmNod_HypOptimization_PCA\(.*\)/\1FFT\2/'`
    new_d=`echo $f|sed -e 's/\(.*\)SmNod_HypOptimization_PCA\(.*\).yml/FFT\2/'`
    cp -v $f $new_f


    # edit files
    sed -i "" -e 's/^\(Name: ".*\)PCA\(.*\)/\1FFT\2/' \
        -e 's/^\(Date:\).*/\1 "23.02.2024"/' \
        -e 's/\(transform: \)"pca"/\1"fft"/' \
        -e 's/- "range"/- "list"/' \
        -e 's/- \[100, 1000, 100\]/- [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]/' \
        -e "s/OutputDirectory:.*/OutputDirectory: \"Data\/Reservoir\/KS_$new_d\/\"/" \
        $new_f
done
