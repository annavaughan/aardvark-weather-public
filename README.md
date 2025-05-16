# Aardvark Weather

This repo contains code and weights used to run the Aardvark Weather model (https://www.nature.com/articles/s41586-025-08897-0). We provide sample data, model weights and code to demonstrate how the trained model is run to produce forecasts. 

## Model weights
Weights for the trained model presented in the paper are provided at `https://huggingface.co/datasets/av555/trained_model/`.
Within this folder there are weights for the encoder, decoder and processor and end to end finetuned weights for one day lead time.
For the decoder and end-to-end, we provide weights for both temperature and wind speed. 

## Data
Sample data is provided in `data/sample_data_final.pkl`. In addition we provide normalisation factors used to generate plots of predictions. For a visualisation of what is included in each timeslice of data passed to the model, see notebooks/data_demo.ipynb. For those interested in training their own end-to-end models, we have additionally prepared a machine learning ready dataset for the data sources utilised in the paper available at https://huggingface.co/datasets/av555/aardvark-weather. 

## Forecast notebooks
Notebooks demonstrating producing a forecast using the trained models are included in the `notebooks/` folder.
Aardvark produces multiple modalities of forecasts.
The notebook `forecast_demo.ipynb` demonstrates loading the complete Aardvark Weather model, generating predictions from the sample data and provides visualisations of the output global gridded and station forecasts.
The notebook `e2e_finetune_demo.ipynb` provides a demonstration of generating optimised station forecasts from the sample data using the end to end finetuned model.

## Training
Aardvark weather is trained in multiple stages.
Scripts to train the model are included in `training/`.
Training of the three main modules is handled in `train_module.py`, processor finetuning in `finetune.py` and end-to-end finetuning in `e2e_train.py`.

__Please note:__ the commands and related scripts under the "Encoder," "Processor," and "Decoder" sections below cannot be executed as they depend on local data loading pipelines, setup for the specific training compute infrastructure. The purpose of these training and finetuning commands is to illustrate the salient points of the training process for the purposes of transperancy and for those interested in the details of the training process and not to provide an executable version.

### Encoder
To train the encoder, run
```
bash train_encoder.sh
```

### Processor
To pre-train the processor, we use the code in
```
bash train_processor.sh
```

To finetune the processor, we use the code in
```
bash finetune.sh
```

### Decoder
To train the decoder, we use the code in
```
bash train_decoder.sh
```

### End-to-end
To tune the model end-to-end, we use the code in
```
bash train_e2e.sh
```
## FAQ

#### Can I access the data the model was trained on?
We provide a dataset with observational data from 2007-2019 at 24 hour resolution at https://huggingface.co/datasets/av555/aardvark-weather. We hope that this will allow others to develop their own end-to-end weather models and explore the vast design space of ML architectures for this task. 

#### Can I run the model in real time? 
Unfortunately the datasets used in this initial prototype are not available in real time. We are however in the process of building a fully operational system, Aardvark 2.0. Updates to this will be provided here.  

#### This model is at lower resolution than other AI models. Do you have any plans to improve on this?
We are currently working on a new version of our model which runs at 0.25 degrees for a wider range of pressure levels. 

#### Will further versions of the model and dataset be provided?

Yes, this project is ongoing and we aim to continue updating this dataset and model. Updates currently in progress include the development of a fully operational system with real time data feed, extension of current historical dataset to 2025 and inclusion of further instruments. If you would like to receive updates as new products become available please email av555@cam.ac.uk with subject line 'Aardvark updates'.

#### Can this dataset be used commercially?

The dataset is released under a non-commercial no-derivatives license. This is due to the lisencing inherited from the source datasets.
