[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6ndC2138)

# Society and Electricity Satellite Segmentation: A comparison of UNet with [a select model: TBD]
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Weights and Biases](https://img.shields.io/badge/Weights%20&%20Biases-FFBE00.svg?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![Scikit Learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

The final project for UCI's CS 175: Project in Artificial Intelligence.

Developed by Brown Rice: [Levi Ramirez](https://github.com/Levi-Ramirez), [Shadi Bitaraf](https://github.com/ShadiBitaraf), [Vedaant Patel](https://github.com/Vedaantp), [Benjamin Wong](https://github.com/chiyeon)

## Goal
**Satellite Society and Electricity Segmentation** targets semantic segmentation, seeking to adapt & use multiple models to achieve high accuracy classification. We will compare the performance of select models against a base UNet model. The model we want to compare UNet to is still in production. However, we expect it to be either UNet3+ or DeepLabV3.

## Instillation
 > TODO PR2

## Getting Started
> TODO PR2: check sample sent, include details on how to run Unet + how to adjust parameters

## Models 
### UNet
This model uses what is called a "skip connection", these are inspired by the nonlinear nature of brains, and are generally good at helping models "remember" informatiion that might have been lost as the network gets longer. These are done by saving the partial outputs of the networks, known as residuals, and appending them later to later partial outputs of the network. In our case, we have the output of the inc layer, as the first residual, and each layer but the last one as the rest of the residuals. Each residual and current partial output are then fed to the Decoder layer, which performs a reverse convolution (ConvTranspose2d) on the partial output, concatenates it to the residual and then performs another convolution. At the end, we end up with an output of the same resolution as the input, so we must MaxPool2d in order to make it the same resolution as our target mask.

![UNet](assets/unet.png)


### [ model for comparison TBD ]
> Here we will write an in depth description of why we chose our particular model, listing its strengths and how well it works with the dataset. If the model was take from somewhere, we would credit the original authors & also note any changes we made here. This model will likely be UNet3+ or DeepLabV3.

## Performance
### UNet
The UNet model performed well and achieved an accuracy high of 67% on the validation set. For the runs 1-4, Sentinel 1 and Sentinel 2 bands were used. For run 5, Sentinel 1, Sentinel 2, VIIRS, and VIIRS MAX Projection were used. For run 6, all bands were used.

> TODO need to import pictures of the predicted values.

Note: The F1 score was set to be logged later on in the sweeps that were ran, so some of the runs do not include an F1 score.

| Epochs | F1 Score | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss | Embedding Size | In Channels | Learning Rate | N Encoders |
| ------ | -------- | ----------------- | ------------- | ------------------- | --------------- | -------------- | ----------- | ------------- | ---------- |
| 5 | NA | 0.59 | 0.84 | 0.56 | 0.90 | 128 | 56 | 0.002917 | 5 |
| 5 | NA | 0.65 | 0.8 | 0.61 | 0.98 | 64 | 56 | 0.07799 | 4 |
| 10 | NA | 0.67 | 0.77 | 0.63 | 0.88 | 32 | 56 | 0.0529 | 5 |
| 10 | NA | 0.68 | 0.75 | 0.64 | 0.88 | 256 | 56 | 0.08564 | 5 |
| 25 | 0.58 | 0.66 | 0.79 | 0.62 | 0.91 | 64 | 66 | 0.08762 | 4 |
| 25 | 0.66 | 0.74 | 0.59 | 0.67 | 0.76 | 256 | 99 | 0.0005 | 5 |


### [selected model: TBD]
> We can include & compare to as many models as need be. 
> Here we would do the same as we did above for UNet, but for the model we choose to select for training and comparision (which is tbd in PR3) 

| Epochs | F1 Score | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| ------ | -------- | ----------------- | ------------- | ------------------- | --------------- |
| 10 | 0.0 | 0.0 | 0 | 0.0 | 0 |
| 20 | 0.0 | 0.0 | 0 | 0.0 | 0 |
| 30 | 0.0 | 0.0 | 0 | 0.0 | 0 |

## Data Set
> TODO PR2: Explain the data set with sample images. Explain the satellites

## Liscense
> TODO PR2:

## Contributers
> TODO PR2

## References
> Todo PR2:
- **[ link to model ]**
- **Acknowledgement of CS175 Staff's HW3 Base Code**
