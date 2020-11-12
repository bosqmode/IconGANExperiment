# Icon GAN Experiment

Experiment to make drawing icons easier by copying a specific style,
in my case: copying the style of icons of the great games: WC3/World of Warcraft

![Intro](https://i.imgur.com/VstfB3A.png)

## Requirements

Python 3.7.5
numpy == 1.17.3
Pillow >= 6.2.1
opencv-python >= 4.1.0
argparse == 1.4.0
tensorboard >= 2.3.0
tensorflow == 2.3.0

``` pip3 install -U -r requirements.txt ```

## Usage

Install the requirements mentioned above.


### Training
1. Find a set of images to copy the style from

2. Run EdgeDetect.py with a parameter '-dir' pointing to the directory the dataset will be made of
```python EdgeDetect.py -dir=PATH_TO_IMAGES```
This will provide two directories filled with images: 'targets' that are rotated versions of the original images
and 'edges' that are "drawing-like" black and white edges of the images.

3. Run GANIt.py with a parameter '--train' to start training
```python GANIt.py --train```

4. Launch Tensorboard to observe the training
```tensorboard --logdir "logs/log"```

Once happy with the results one can stop the training
and proceed trying out the drawing!

In my case it took around 8000 steps to get reasonable results.

### Drawing

Run GANIt.py with a parameter '--load_weights' to load the weights created in training
```python GANIt.py --load_weights```

(Note: loading weights can also be used to continue training, the saved weights will be named 'G_Weights' and 'D_Weights' for the Generator and Discriminator respectively)

## Results

The results of this experiment were mediocre. While the network seems to output reasonable results while
training, in reality, when drawing by hand, the outputted images will often look like nothing that one may want.
Reasons for this might be the way the edges are calculated, often the edges will look similar to each other
which can confuse the network, and the fact that the automatically detected edges are usually not quite
like how a human would actually draw. To correct this, one should probably have a human to provide the real drawn edges
to correlate better to how human's actually draw.

![Drawings](https://i.imgur.com/mPLKVED.jpg)

## Acknowledgments

* https://arxiv.org/pdf/1611.07004.pdf
* https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

