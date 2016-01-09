# ImageNet Experiments

## Installation

In this repository, we use a self-brewed caffe, which provides Batch Normalization and faster MPI-based parallel training. In case you don't need such features, you may replace it with your own caffe.

The detailed installation instructions of the self-brewed caffe can be found [here](https://github.com/yjxiong/caffe).

## Make a low-rank net

You can use `lowrank_approx.py` to make a low-rank net. For example,

    python2 lowrank_approx.py \
      --model models_vgg/vgg_deploy.prototxt \
      --config models_vgg/config.json \
      --save_model models_vgg/vgg_lowrank_deploy.prototxt

This will make a low-rank VGGNet prototxt only.

You can also download some pretrained models

+  [CaffeNet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel)
+  [VGGNet](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)

And use the tool to convert them to low-rank ones

    python2 lowrank_approx.py \
      --model models_vggnet/vggnet_deploy.prototxt \
      --config models_vggnet/config.json \
      --save_model models_vggnet/vggnet_lowrank_deploy.prototxt \
      --weights VGG_ILSVRC_16_layers.caffemodel \
      --save_weights vggnet_lowrank.caffemodel