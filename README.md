# dogcat

Build CNN using tensorflow to tell wether a image contains dog or cat.

# Environment

Python3.5, tensorflow-1.0 (sudo pip3 install tensorflow)

# Dataset

https://www.kaggle.com/c/dogs-vs-cats/data

https://www.kaggle.com/c/dogs-vs-cats/download/test1.zip

https://www.kaggle.com/c/dogs-vs-cats/download/train.zip

# Preprocess

Use `python3 preprocess.py` to convert raw image to n x n in gray.

# Train

```
FNAME=cnn.input64.conv3
python3 ${FNAME}.py > log/${FNAME}.log 2>&1 &; tail -f log/${FNAME}.log
```