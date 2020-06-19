# Yolo_V4

#### Compiling Darknet

Compile the darknet with `make` command after editing the `Makefile`. Edit the following lines with respect to your system configuration,put 1 instead of 0 if you have these thing.`GPU=0 CUDNN=0 CUDNN_HALF=0 OPENCV=0`.

#### Pre-trained Weights

Download the pretrained weigths for convolutional layers and put to the directory `cfg/yolov4.conv.137`. [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )

#### Create Dataset

1. YOLO takes .txt file annotations for model training but mostly we have .xml annotations, so we have a script that convert .xml file annotations to the required one.

2. Put your dataset into the data folder. Keep in mind naming and directory structure is very important here. Put `annotations` and `images` in data folder. `annotations` should contain all xml file and `images` should contain all the images.

3. Edit the `classes.txt` and put your label names there. e.g. if you have 3 classes person,dog and cat
```
person
dog
cat
```

4. run `./create_data.sh` in terminal. if it gives error, first execture `chmod +x create_data.sh`.

5. Edit the `obj.data` file as it is one of the input during training.
```
classes= 80
train  = data/train_files.txt
valid  = data/train_files.txt
names = data/classes.txt
backup = backup/
```


#### Setting cfg file

1. Create file `yolo-obj.cfg` with the same content as in `yolov4-custom.cfg` (or copy `yolov4-custom.cfg` to `yolo-obj.cfg)` and:

  * change line batch to `batch=64` or whatever you think is suitable
  * change line subdivisions to `subdivisions=16` or whatever you think is suitable.
  * change line max_batches to `classes*2000`. e.g. `max_batches=6000` if you have 3 classes.
  * change line steps to 80% and 90% of max_batches, e.g. [`steps=4800,5400`]  
  * set network size `width=416 height=416` or any value multiple of 32.
  * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers.
  * change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers.

  So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.


#### Training

run `.train.sh` and training will start.

   * (file `yolo-obj_last.weights` will be saved to the `backup\` for each 100 iterations)
   * (file `yolo-obj_xxxx.weights` will be saved to the `backup\` for each 1000 iterations)

   * If stopped training at a particular point you could always restart training from there by making minor change to the `train.sh` file. Give the previous saved weights instead of the pretrained weight. and execute the script again.
   ```
   ./darknet detector train data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_xxxx.weights -dont_show
   ```

#### Inference

run `python3 Image_inference.py -i img.jpb -o output.jpg` for inference of a single image.

#### Issue

1. if faced with opencv issue while compiling with `make` command use `sudo apt install libopencv-dev`. It will solve the opencv error.