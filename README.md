[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6ndC2138)

# [ Project Name ]
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Weights and Biases](https://img.shields.io/badge/Weights%20&%20Biases-FFBE00.svg?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![Scikit Learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

The final project for UCI's CS 175: Project in Artificial Intelligence.

Developed by Brown Rice: [Levi Ramirez](https://github.com/Levi-Ramirez), [Shadi Bitaraf](https://github.com/ShadiBitaraf), [Vedaant Patel](https://github.com/Vedaantp), [Benjamin Wong](https://github.com/chiyeon)

## Goal
**[ Project Name ]** targets semantic segmentation, seeking to adapt & use [ model ] to achieve high accuracy classification. We will compare the performance of **[ model ]** against several other models including **(temporary: )** UNet with pretrained weights, UNet, Resnet, and Segmentation CNN.

## [ Model ]
> Here we would write an in depth description of why we chose our particular model, listing its strengths and how well it works with the dataset. If the model was take from somewhere, we would credit the original authors & also note any changes we made here.

## Performance
### [ model ]
> This is where we would put a brief description of the chosen model & how it works, as well as its overall performance. We could optionally also include the hyperparameters that lead us to these particular results.

| Epochs | F1 Score | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| ------ | -------- | ----------------- | ------------- | ------------------- | --------------- |
| 10 | 0.0 | 0.0 | 0 | 0.0 | 0 |
| 20 | 0.0 | 0.0 | 0 | 0.0 | 0 |
| 30 | 0.0 | 0.0 | 0 | 0.0 | 0 |

### UNet
UNet is a network where the input is downscaled down to a lower resolution with a higher amount of channels, but the residual images between encoders are saved to be concatednated to later stages, creatin the nominal "U" shape. The highest validation accuracy that was achieved is 67% and the highest F1 score achieved is 0.66.

Note: The F1 score was set to be logged later on in the sweeps that were ran, so some of the runs do not include an F1 score.

| Epochs | F1 Score | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| ------ | -------- | ----------------- | ------------- | ------------------- | --------------- |
| 5 | NA | 0.59 | 0.84 | 0.56 | 0.90 |
| 5 | NA | 0.65 | 0.8 | 0.61 | 0.98 |
| 10 | NA | 0.67 | 0.77 | 0.62 | 0.88 |
| 10 | NA | 0.68 | 0.75 | 0.64 | 0.88 |
| 25 | 0.58 | 0.66 | 0.79 | 0.62 | 0.91 |
| 25 | 0.66 | 0.74 | 0.59 | 0.67 | 0.76 |

## References
- **[ link to model ]**
- **Acknowledgement of CS175 Staff's HW3 Base Code**
