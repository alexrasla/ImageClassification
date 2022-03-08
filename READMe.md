# Running Program
## Training Model

In order to train an image classification model using my program, first connect to the Google Colab GPU by 
navigating to www.research.colab.com and clone this repository.

Next, to edit the model type and its hyperparameters, change the values in the config.py. Finally, execute: 

```
!python3 train.py
```

to train the image classification model specificied in the MODEL variable in the Config class.

## Evaluating Model
Once this model is trained, evaluation can be performed by executing: 
``` 
!python3 eval.py --model [path to model]
``` 
Example:
```
!python3 ./ImageClassification/eval.py --model ./drive/MyDrive/checkpoint.pth
```
                 
The evaluation is performed on the CIFAR-10 testset, and the program generates and saves a 
confusion matrix, which is used to plot and analyze the results using 
```
!python3 plot.py  --dir [path to model directory]
``` 

Example:
```
!python3 plot.py  --dir ./models/batch_norm
``` 