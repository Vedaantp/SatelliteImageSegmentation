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

### Step 0, RECOMMENDED: Set up Virtual Project Environment
To keep the build clean, we recommend using a virtual environment in order to have a reproducible and stable envrionment.

1. Create a virtual environment:
   
   `python3 -m venv esdenv`
2. Activate the virtual environment:
   * On macOS and Linux:
  
        `source esdenv/bin/activate`
   * On Windows:
  
        `.\esdenv\Scripts\activate`

To deactivate the virtual environment, type `deactivate`.

### Step 1: Set up Virtual Project Environment
Install the required packages:
    `pip install -r requirements.txt`
### Step 2: 
PUT INFO FOR LOGGING INTO WANDB
Can use Wandb for experiment tracking, visualization, and collaboration in this project. Follow this page for [logging in](https://wandb.auth0.com/login?state=hKFo2SB4VS1WN2dXa0k4OHhTYndvelBiOGRMckRUWl9feGJ5VaFupWxvZ2luo3RpZNkgYTVDY0lUcXBPSVJsUVNSOXhWOTFMenpsRnZTcFBWUEajY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=TmpIZ2NwflJqWVFCT0VvMA%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email).
1. run `wandb login`
2. Input W&B API key into the prompt. If you don't have an account, you'll need to sign up first on their website. Once you've logged in and authenticated your account, you can start using 
3. Input your project name associated with your account in train.py with the line wandb.init(project="PROJECT_NAME"), replacing PROJECT_NAME with the name of your project.


### Step 3:
Download the dataset, place it into a directory. Put it in a directory called `data/raw`. The full path after download and placing the data here should be `data/raw/Train`.

Now you should be ready to run the commands to run the models to train on this dataset. Look at the Training section below to see command to train

## Models 
### UNet
This model uses what is called a "skip connection", these are inspired by the nonlinear nature of brains, and are generally good at helping models "remember" informatiion that might have been lost as the network gets longer. These are done by saving the partial outputs of the networks, known as residuals, and appending them later to later partial outputs of the network. In our case, we have the output of the inc layer, as the first residual, and each layer but the last one as the rest of the residuals. Each residual and current partial output are then fed to the Decoder layer, which performs a reverse convolution (ConvTranspose2d) on the partial output, concatenates it to the residual and then performs another convolution. At the end, we end up with an output of the same resolution as the input, so we must MaxPool2d in order to make it the same resolution as our target mask.

![UNet](assets/unet.png)


### [ model for comparison TBD ]
> Here we will write an in depth description of why we chose our particular model, listing its strengths and how well it works with the dataset. If the model was take from somewhere, we would credit the original authors & also note any changes we made here. This model will likely be UNet3+ or DeepLabV3.

## Performance
### UNet
> TODO PR2: place training results as well as sample segmentation images. Write a brief descriptoin of results. Also include the training parameters + bands/satellites used.

| Epochs | F1 Score | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| ------ | -------- | ----------------- | ------------- | ------------------- | --------------- |
| 10 | 0.0 | 0.0 | 0 | 0.0 | 0 |
| 20 | 0.0 | 0.0 | 0 | 0.0 | 0 |
| 30 | 0.0 | 0.0 | 0 | 0.0 | 0 |


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

## Training
We will train the models using the model architectures defined above in conjunction with the PyTorch Lightning Module for ease of running the training step in `train.py.` To monitor model training make sure to make an account with Weights and Biases for yourself and then create a team. For details on how to get started see [How to Use W&B Teams For Your University Machine Learning Projects for Free](https://wandb.ai/ivangoncharov/wandb-teams-for-students/reports/How-to-Use-W-B-Teams-For-Your-University-Machine-Learning-Projects-For-Free---VmlldzoxMjk1Mjkx).

### `ESDConfig` Python Dataclass
In `src/utilities.py` we have created an `ESDConfig` dataclass to store all the paths and parameters for experimenting with the training step. If you notice, in the main function of `scripts/train.py`, `scripts/evaluate.py`, and `scripts/train_sweeps.py` we have provided you with code that utilize the library `argparse` which takes command line arguments using custom flags that allow the user to overwrite the default configurations defined in the dataclass we provided. When running train, for example, if you would like to run training for the architecture `SegmentationCNN` for five epochs you would run:

`python -m scripts.train --model_type=SegmentationCNN --max_epochs=5`

Here is more information on [`argparse`](https://docs.python.org/3/howto/argparse.html).

### Hyperparameter Sweeps
We will be using Weights and Biases Sweeps by configuring a yaml file called `sweeps.yml` in order to automate hyperparameter search over metrics such as batch size, epochs, learning rate, and optimizer. You may also experiment with the number of encoders and decoders you would like to add to your model architecture given that you are sensitive to the dimensions of your input image and the dimensions of the output prediction with respect to the ground truth. Some useful articles on how to perform sweeps and use the information to choose the best hyperparameter settings for your model can be found:
- [Tune Hyperparameters](https://docs.wandb.ai/guides/sweeps)
- [Running Hyperparameter Sweeps to Pick the Best Model](https://wandb.ai/wandb_fc/articles/reports/Running-Hyperparameter-Sweeps-to-Pick-the-Best-Model--Vmlldzo1NDQ0OTIy)

To run training with the hyperparameter sweeps you define in `sweeps.yml` please run `train_sweeps.py --sweep_file=sweeps.yml` provided for you.

## Validation
You will run validation using the script `evaluate.py` where you will load the model weights from the last checkpoint and make a forward pass through your model in order to generate prediction masks. Use `ESDConfig` dataclass in `utilities.py` to set the default configuration for the validation loop when arguments are not passed via command line.

### Visualization: Restitching Predictions to Compare with Ground Truth
`evaluate.py` calls functions in `visualization/restitch_plot.py` which restitches the predicted subtiles and groundtruth back to the original 16x16 dimensions for plotting purposes using the same colormap as defined in the IEEE GRSS ESD 2021 Competition.


## Liscense
> TODO PR2:

## Contributers
> TODO PR2

## References
> Todo PR2:
- **[ link to model ]**
- **Acknowledgement of CS175 Staff's HW3 Base Code**
