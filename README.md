# Azure Machine Learning Engineer - Capstone Project to predict the Heart Disease Status applying MLOps

Azure the cloud solution from microsoft offers the well-known advantages of the vertical scaling of computing power, the provision of web services and the use of existing cloud solutions to create the best products. <br/> That project uses these advantages for training the machine learning model with AutoML or hyperparameter tuning using Hyperdrive from a generated database to perform classification. Compute instances are created and made available in the cloud. The best trained model is deployed as a webservice whose endpoint API can be consumed via REST method to obtain the model's prediction. Key benefit of using the Python SDK, Jupyther Notebook leads to an elegant and reuseable way of automating the Machine learning task from cleaning data, training models, deploying the best model as a REST API and testing it's endpoint. <br/>
A more detailed *Project Workflow* ant the *Project Implementation* focus the goal of the project next to the technical details. <br/> <br/>
![Workflow](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Images/workflow.PNG)

## Dataset
Despite the scientific advancement over the past decades, there are still numerous research areas in the field of heart disease. It is of particular interest to record easy-to-follow parameters and use them to infer a person's heart disease status.

### Overview of the data set
Envestigating the status of the heart disease with machine learning methods requires a large number of labeled trainings and test data. It is therefore advisable to use data from the [kaggle](https://www.kaggle.com/) data platform. In addition to the large number of data records, information is also provided about the distribution of the attributes in a data record, which gives you a first impression of the data. <br/>
[Heart Disease](https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv), a tabular dataset consits of 304 rows, 12 attributes and a labeled target.
An excerpt from the chosen dataset shows the contained attributes, the following form the input vectors of the machine learning model and the labled output`target`:
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

### Task of the machine learning model
The aim of this work is to train a machine learning classification algorithm using the data received from the above mentioned database. Therefore the 12 featured are used as input vectors during training and the machine learning algorithm is learning a mapping from the input vector to the output `target`. After finishing training input data of the same shema as the input vectors are fed into the trained network and the network is going to predict the output related to the input data which should give a pearson the opportunity to get information about the heart disease status.

### Access to the data
Kaggle provided the Heart Disease dataset as a csv file, which is saved on github and uploaded as Raw data form the local file and registered to the Azure Workspace as a tabular dataset. It is also possible to work with data sets in machine learning algorithems.

## Automated ML
At first the task, solving the machine learning problem of the heart disease dataset applying AutoML has to be described. Using AutoML, the key steps have to be described in the AutoMLConfig, which lists automl_setting - these are useful for finetuning the model - the calssification task or attributes like enabling of onnx compatible models.
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/automlsettings.PNG) <br/>


### Results of the AutoML run
The key benefit of AutoML is the variaty of different classification algorithems, their outputs are compared against eachother to find the best fitting model. Here the best fitted model is the VotingEnsemble, a detailed view shows the n_estimators=25, n_jobs=0 and the metrix of weights for more information to look into the model.
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/parameters.PNG) <br/>
More details are just shown in the `RunDetails` widget.
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/widget1.PNG) <br/>
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/widget2.PNG) <br/>
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/widget3.PNG) <br/>
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/widget4.PNG) <br/>



## Hyperparameter Tuning
The first branch of the workflow project leads to receiving a trained model using AutoML. Settings and configurations used for this experiment are a `BanditPolicy` as an early termination policy, `RandomParameterSampling` of the tuneable hyperparameters of the classifier and the `HyperDriveConfig`.
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Hyperdrive/sampling_policy_2hp.PNG) <br/>

`BanditPolicy` as an early termination policy is required to automatically terminate poorly performing runs which imporves computational efficiency. With it's attributes evaluation interval and slack factor Bandit Policy terminates runs in which the primary metric is not within the specified slack factor amound compared to the best performing run and therefore terminates bad runs early. <br/> <br/>
`RandomParameterSampling` with '--C' the inverse of the regularization strength ranged from 0.1 to 1.0 - smaller values specify stronger regularization - and '--max_iter' the maximum amount of iterations taken for the solver to converge in a discrete numbers of 150, 200 and 250. Using Hyperdrive to find the best combinatin of 'C' and 'max_iter' and therefore the best set of hyperparameters for a maximal accuracy is the goal of hyperparameter tuning. Regarding the [LogisticRegression scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model which is used for classificatino it turns out that 'C' and 'max_iter' are good good candidates for tuning, because the algorithm is build to fulfill that task. <br/> <br/>
`HyperDriveConfig` describes and lists the settings and attributes like run configuration where the training script is listed, the choosen hyperparameter sampling, the choosen terminatin policy and the choosen metric. The deployed config will lead to the trained model with best tuned hyperparameters using Azure in combination with Hyperdrive. <br/> 

### Results of Hyperparameter Tuning using Hyperdrive
After completing the traing task, the received model can be found in the experiment section and is listed as Run 1 with the corresponding Run ID.
![comp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Hyperdrive/1.PNG) <br/>
Child runs offer a moe detailed view of the hyperparameter tuning combinations.
![cr](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Hyperdrive/2.PNG) <br/>
As it turned out with an accuracy of 0.918 (even better than the runs before) the hyperparameter combination of the Regularization Strength of 0.248 and 250 Max iterations lead to the best tuned model.
![cr3](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Hyperdrive/3.PNG) <br/>
Using the Python SDK, a detailed overview of the training details is given using the `RunDetails` widget.
![widget](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Hyperdrive/run_details.PNG) <br/>

## Model Deployment
As it turned out, the VotingEnsemble resulted as the best AutoML model with an weighted Accuracy of 0.927, even better than the classification model after applying hpyterparameter tuning using Hyperdrive with an Accuracy of 0.901, which has to be deployed as a webservice. <br/> 
It's notrivial architecture is represented, using [NETRON](https://github.com/lutzroeder/netron) an open source tool for investigating and documenting Machine Learning Models in a pictorial way. <br/>
Following the data flow, the 13 input features are scaled and casted in different ways resulting in an `FeatureVectorizer` which in turn results in the superposition of the 5  `TreeEnsembleClassifiers` and the `LinearClassifier` which each one is voted by a `Scaler`. Each `label_out` gets the associated `probabilities_out` of the VotingEnsemble as model output. Due to the Classification task of the deployed model, the output `label_out` is used as response.
![hyperdrive](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Images/best_AutoMLmodel.png) <br/>
The VotingEnsemble is deployed as a Azure Container Instance (ACI) the resulting webservice is a HTTP endpoint with load balancing and a REST-API. Data can be send to the API as a POST request of a JSON object and receive the deployed models prediction as a response. Using the [swagger.json](https://github.com/Daniel-car1/nd00333-capstone/blob/main/swagger_nice_style.json), detailed information about the API schemes like the consumed and produced data, methods, responses or example input data can be taken into account, which follow the same style as the dataset. 
![hyperdrive](https://github.com/Daniel-car1/nd00333-capstone/blob/main/Images/swagger_nice_style.PNG) <br/> <br/>
![deploy](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/model_deployment.PNG) <br/>
Deploying an registered model requires the following attributes:
* `workspace`
* `name` of the webservice/endpoint
* `models` which should be deployed
* `inference_config` using the entry_script and the environment of the model which should be deployed
* `deployment_config`, describes the resources like cpu_cores and memory_gb, and attributes like auth_enabled or enable_app_insights <br/>
![state](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/state.PNG) <br/>
For communication with the webservice the URI is at least required, the Swagger JSON gives detailed information about the schema to use.
![state](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/Endpoint_url_application%20insight.PNG) <br/>
After successfully deploying the model with the healthy webservice state, the endpoint can be consumed sending a POST request to the Scoring URI with the JSON object in the body and Content-Type and Authorization int the headers. As a consequence a JSON response is received from the webservice. <br/>
![resp](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/response.PNG) <br/>
The result of the choosen request ist [1] which indicates 'asymptomatic' of heart disease status and reflectes the expected result. <br/>

## Screen Recording on Youtube
[Azure Machine Learning Engineer - Capstone Project to predict Heart Disease Status applying MLOps](https://www.youtube.com/watch?v=2bqdOrTVWL8)

## Standout Suggestions
The suggested Standout Suggestions *Convert the best model to ONNX format* and *Enable logging in the deployed web app* were executed in a detailed way. <br/>
* *Converting the best model to ONNX format* requires preparatino in the `AutoMLConfig`, setting `enable_onnx_compatible_models = True`. Using `OnnxConverter` the best model is saved as an [ONNX model](https://github.com/Daniel-car1/nd00333-capstone/blob/main/best_model.onnx), which can be deployed in Azure, on Windows devices and even on iOS devices. The received model is mentioned and documented in the Model Deployment section.
* *Enable logging in the deployed web app* is implemented using the Python SDK code snippet in the Jupyter Notebook. Detailed information about the endpoint's characteristics are `Failed requests`= 0, `Server response time`= 0.78 ms and the number of `Server requests`= 9.
![log0](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/application_insight.PNG) <br/> 
![log0](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/logs_0.PNG) <br/> 
![log](https://github.com/Daniel-car1/nd00333-capstone/blob/main/AutoML/logs.PNG) <br/> 

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
