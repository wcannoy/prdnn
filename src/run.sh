#!/bin/bash

python dnn.py dense 0.01

keep_ratios=(0.2 0.1 0.05 0.01)
methods=(isnip igrasp synflow fsnip fgrasp fsynflow)

if [ ! -d "../plots/cache" ]; then
    mkdir ../plots/cache
fi

for method in ${methods[@]};do
    for ratio in ${keep_ratios[@]};do
        python dnn.py ${method} ${ratio}
    done
done
