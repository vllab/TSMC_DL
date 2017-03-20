# DCGAN in Tensorflow

## Dataset
We test this code on `celebA`.
- Download celebA dataset
(use the [donwload.py](download.py) script borrowed from [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow/download.py))
```bash
python download.py celebA
```

## Training
We offer two strategies to train the DCGAN model.
- Strategy 1 (use the default settings)
```bash
python main.py 
```
- Strategy 2
```bash
python main.py --training_strategy=2 --g_step=2 
```

## Testing
Not implement yet. Leave for you to exercise.

## Training Process Visualization
- Strategy1

  ![Strategy1](strategy1_training.gif)

- Strategy2

  ![Strategy2](strategy2_training.gif)
