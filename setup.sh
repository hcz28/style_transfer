#! /bin/bash

mkdir data checkpoints 
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P data/
wget http://images.cocodataset.org/zips/train2014.zip -P data/
unzip data/train2014.zip -d data/

mkdir data/style_targets
wget
http://bpic.588ku.com/element_origin_min_pic/16/12/22/c903ceed25128f30592f101717ef656f.jpg
-P data/style_targets

mv data/style_targets/*.jpg data/style_targets/plum_blossom.jpg

python style.py --style data/style_targets/plum_blossom.jpg
