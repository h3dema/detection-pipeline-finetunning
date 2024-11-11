# Models


This folder contains the definitions of all models implemented in this library.
Open `create_fasterrcnn_model.py` to see the available models.
This file also shows an example (on main body) of how to instantiate a model.

**Notice** that `create_model` does not provide flexibility on the creation of the architecture. If you need a custom one, you will need to create a python file with it in this folder. `fasterrcnn_vgg16.py` shows an example of a simple structure to create the model.