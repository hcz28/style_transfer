#! /bin/bash

mkdir data checkpoints 
cd data
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

mkdir style_targets
wget http://bpic.588ku.com/element_origin_min_pic/16/12/22/c903ceed25128f30592f101717ef656f.jpg
mv *.jpg plum_blossom.jpg

python style.py --style /data/style_targets/plum_blossom.jpg
