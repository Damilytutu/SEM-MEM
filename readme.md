##SMEM
This repository has the source code for the paper "An Improved LSTM-based Network: Learning Explicit Shape and Motion Evolution Maps for Skeleton-based Human Action Revognition"


##Dependencies

- Theano
- Keras
- Scipy
- matplotlib
- Numpy

##Source Code Description

We implement our network based on Keras. Keras supports custom operation, so several novel layers proposed in our paper can be easily implemented. Also, the LSTM-based architecture of our method can be easily implemented with Keras Functional API.  Some files are described as follows.

- `net.py` provides the code for overall fusion model (SMEM)
- `data_shape.py` provides the code for shape evolution maps (SEM)
- `data_motion.py` provides the code for motion evolution maps (MEM)
- `main.py` provides the code for main exe file


others:
kutilities : provides the code for weighted aggregate layer (WAL), needing compile the setup.py to setup
mul: provides the fusion model (SMEM)
mul_WAL: provides the fusion model with WAL (SMEM + WAL)

about how to obtain the SEM and MEM, you can easily implement it.

## our experimental NTU RGB+D dataset's skeleton data, you can download at: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp