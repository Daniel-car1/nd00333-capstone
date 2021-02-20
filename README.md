*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project. 

ML Ops, Classification, Data prparation, AutoML, Hyperparamter tuning Hyperdrive, investigate machine larning models, automation, Jupyther Notebook, deploy model as webservice, prediction, consume endpoint's REST API, investigate endpoints

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset
https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv <br/>
[Heart Disease](https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv) <br/>
An excerpt from the chosen dataset shows the contained attributes, the following form the input vectors of the machine learning model, except `target`:
* `age` in years
* `sex` (1 = male; 0 = female)
* `cp` chest pain type
* `trestbps` resting blood pressure 
* `chol` serum cholestoral in mg/dl
* `fbs` fasting blood sugar
* `restecg` resting electrocadiographic results
* `thalach` maximum heart rate achieved
* `exang` exercise induced angina (1 = yes; 0 = no)
* `oldpeak`ST depression induced by exercise relative to rest
* `slope` of the peak exercise ST segment
* `ca` number of major vessels (0-3) colored by flourosopy
* `thal` (0 = normal; 1 = fixed defect; 2 = reversable defect)

* `target` indicates the heard disease status (0 = heart disease; 1 = asymptomatic) and is the label of the dataset which has to be predicted and send as a response from the deployed model.

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input. <br/>
As it turned out, the VotingEnsemble resulted as the best AutoML model, which has to be deployed as a model endpoint. <br/> 
It's notrivial architecture is represented, using [NETRON](https://github.com/lutzroeder/netron) an open source tool for investigating and documenting Machine Learning Models in a pictorial way. <br/>
Following the data flow, the 13 input features are scaled and casted in different ways resulting in an `FeatureVectorizer` which in turn results in the superposition of the 5  `TreeEnsembleClassifiers` and the `LinearClassifier` which each one is voted by a `Scaler`. Each `label_out` gets the associated `probabilities_out` of the VotingEnsemble as model output. Due to the Classification task of the deployed model, the output `label_out` is used as response.
![hyperdrive](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Images/best_AutoMLmodel.png) <br/>
The VotingEnsemble is deployed as a Azure Container Instance (ACI) the resulting webservice is a HTTP endpoint with load balancing and a REST-API. Data can be send to the API as a POST request of a JSON object and receive the deployed models prediction as a response. Using the [swagger.json](https://github.com/Daniel-car1/nd00333-capstone/blob/main/swagger_nice_style.json), detailed information about the API schemes like the consumed and produced data, methods, responses or example input data can be taken into account, which follow the same style as the dataset. 
![hyperdrive](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Images/swagger_nice_style.PNG) <br/>
Deploying an registered model requires the following attributes:
* `workspace`
* `name` of the webservice/endpoint
* `models` which should be deployed
* `inference_config` using the entry_script and the environment of the model which should be deployed
* `deployment_config`, describes the resources like cpu_cores and memory_gb, and attributes like auth_enabled or enable_app_insights <br/>

#BILD <br/>
After successfully deploying the model with the healthy webservice state, the endpoint can be consumed sending a POST request to the Scoring URI with the JSON object in the body and Content-Type and Authorization int the headers. As a consequence an response is received from the webservice. <br/>
#BILD <br/>
The result of the choosen request ist [1] which indicates an ........ of heart .... <br/>
## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted. <br/>
The suggested Standout Suggestions *Convert the best model to ONNX format* and *Enable logging in the deployed web app* were executed in a detailed way. <br/>
* *Converting the best model to ONNX format* requires preparatino in the `AutoMLConfig`, setting `enable_onnx_compatible_models = True`. Using `OnnxConverter` the best model is saved as an [ONNX model](https://github.com/Daniel-car1/nd00333-capstone/blob/main/best_model.onnx), which can be deployed in Azure, on Windows devices and even on iOS devices. <br/> #BILD
* *Enable logging in the deployed web app* is implemented using the Python SDK code snippet in the Jupyter Notebook. Detailed information about the endpoint's characteristics are `Failed requests`=XXX, `Server response time`= XX ms and the number of `Server requests`= XX.
<br/> #BILD
<br/> #BILD

Nevertheless, expanding that project is even possible by
* deploying the model of the Edge using Azure IoT Edge
* reduce the input vector ot the model to accelerate the training and reduce the number of nodes to save costs

## Sources of knowledge
[Udacity - Machine Learning Engineer for Microsoft Azure](https://www.udacity.com/course/machine-learning-engineer-for-microsoft-azure-nanodegree--nd00333) <br/>
[Azure Container/Kubernetes](https://azure.microsoft.com/de-de/services/kubernetes-service/) <br/>
[Azure Machine Learning Operations (MLOps) - Pipelines - Automation](https://azure.microsoft.com/de-de/services/machine-learning/mlops/) <br/>
[Azure Machine Learning documentation](https://docs.microsoft.com/en-us/azure/machine-learning/) <br/>
[Azure AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azure-ml-py) <br/>
[Azure HyperDrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?view=azure-ml-py) <br/>
[Azure consume and deploy model endpoint](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python) <br/>
[ONNX](https://onnx.ai/) <br/>
