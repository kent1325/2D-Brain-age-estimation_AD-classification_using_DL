# 2D Brain Age Estimation and Alzheimer’s disease Classification using Deep Learning

This project aims to apply brain age as a biomaker for Alzheimer's disease. Thus, the GLobal-Local Transformer from [@shengfly](https://github.com/shengfly/global-local-transformer) to estimate the brain age using MR images. Additionally, we use the learned feature vector from the transformer as input to a multi-layer perceptron (MLP) model do Alzheimer’s disease diagnosis.




## Authors

- [@EmilBuch](https://www.github.com/EmilBuch)
- [@kent1325](https://github.com/kent1325)
- [@Kowalski2332](https://github.com/Kowalski2332)


## Documentation

We have split the project into two main pieces. The python file `main.py` performs five-fold cross validation, training and test of the transformer.

The python file `classifier_validation.py` performs the Alzheimer's disease diagnosis.

