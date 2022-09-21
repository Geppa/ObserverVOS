# ObserverVOS
Observer Network in VOS Reaserch in Augmented Laboratory


<br>


### Model Directory 

````md
├─ObserverVOS/
│  │
│  ├──model/                              <- networks
│  │      ├── observer.py                 
│  │      ├── stm_model.py
│  │
│  ├──Datasets/                           <- loading data
│  │      ├── obstm_dataset.py            <- Davis dataset
│  │      ├── seg_transfo.py              <- adapt pytorch data augmentation for segmentation
│  │
│  ├──Utils/
│  │      ├── adv_attacks.py              <- fct advesarial attacks
│  │      ├── affichange.py               <- fct for plot 
│  │      ├── conv1x1.py                  <- fct for 1x1 convolution layer
│  │      ├── loss.py                     <- focal loss
│  │      ├── metrics.py                  <- metrics
│  │      ├── utils.py                    <- useful functions 
│  │
│  ├──README.md                           <- this 
│  ├──evaluation.py                       <- test, evaluation 
│  ├──train.py                            <- train
│  ├──main.py                             <- main
│  ├──helpers.py                          <- fct for STM
│  ├──requirements.txt                    <- requirements
│  ├──config.yaml                         <- config

````


<br>
