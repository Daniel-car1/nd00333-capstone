import json
import os
import traceback

from azureml.core.workspace import Workspace
from azureml.train.automl._remote_script import featurization_wrapper

args = sys.argv
run_id = "%%RUN_ID%%"
aml_token = None
script_directory = None
print("Starting the featurization....")

task_type = "classification"  # PLACEHOLDER
entry_point = "get_data.py"  # PLACEHOLDER
preprocess = "True"  # PLACEHOLDER
dataprep_json = "{\"training_data\": \"{\\\"blocks\\\": [{\\\"id\\\": \\\"17be2681-4f36-475c-a446-a1c60d9a2778\\\", \\\"type\\\": \\\"Microsoft.DPrep.GetFilesBlock\\\", \\\"arguments\\\": {\\\"isArchive\\\": false, \\\"path\\\": {\\\"target\\\": 4, \\\"resourceDetails\\\": [{\\\"path\\\": \\\"https://raw.githubusercontent.com/Daniel-car1/nd00333-capstone/main/heart.csv\\\"}]}}, \\\"localData\\\": {}, \\\"isEnabled\\\": true, \\\"name\\\": null, \\\"annotation\\\": null}, {\\\"id\\\": \\\"21d62914-ab3b-418f-a722-08fb371addb0\\\", \\\"type\\\": \\\"Microsoft.DPrep.ParseDelimitedBlock\\\", \\\"arguments\\\": {\\\"columnHeadersMode\\\": 3, \\\"fileEncoding\\\": 0, \\\"handleQuotedLineBreaks\\\": false, \\\"preview\\\": false, \\\"separator\\\": \\\",\\\", \\\"skipRows\\\": 0, \\\"skipRowsMode\\\": 0}, \\\"localData\\\": {}, \\\"isEnabled\\\": true, \\\"name\\\": null, \\\"annotation\\\": null}, {\\\"id\\\": \\\"77c7020a-fc3e-4063-a97a-3e45dfaabbda\\\", \\\"type\\\": \\\"Microsoft.DPrep.DropColumnsBlock\\\", \\\"arguments\\\": {\\\"columns\\\": {\\\"type\\\": 0, \\\"details\\\": {\\\"selectedColumns\\\": [\\\"Path\\\"]}}}, \\\"localData\\\": {}, \\\"isEnabled\\\": true, \\\"name\\\": null, \\\"annotation\\\": null}, {\\\"id\\\": \\\"f0081254-54be-4888-9887-7bb11e267e51\\\", \\\"type\\\": \\\"Microsoft.DPrep.SetColumnTypesBlock\\\", \\\"arguments\\\": {\\\"columnConversion\\\": [{\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"age\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"sex\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"cp\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"trestbps\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"chol\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"fbs\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"restecg\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"thalach\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"exang\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"oldpeak\\\"}}, \\\"typeProperty\\\": 3}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"slope\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"ca\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"thal\\\"}}, \\\"typeProperty\\\": 2}, {\\\"column\\\": {\\\"type\\\": 2, \\\"details\\\": {\\\"selectedColumn\\\": \\\"target\\\"}}, \\\"typeProperty\\\": 2}]}, \\\"localData\\\": {}, \\\"isEnabled\\\": true, \\\"name\\\": null, \\\"annotation\\\": null}], \\\"inspectors\\\": [], \\\"meta\\\": {\\\"savedDatasetId\\\": \\\"c558cab2-0772-41e8-a424-7a01c66cc610\\\", \\\"datasetType\\\": \\\"tabular\\\", \\\"subscriptionId\\\": \\\"d7f39349-a66b-446e-aba6-0053c2cf1c11\\\", \\\"workspaceId\\\": \\\"06ced75c-6ef1-49c3-a6b8-1a1fc8649d8c\\\", \\\"workspaceLocation\\\": \\\"southcentralus\\\"}}\", \"activities\": 0}"  # PLACEHOLDER
num_iterations = None # PLACEHOLDER
enable_subsampling = None # PLACEHOLDER
enable_streaming = False # PLACEHOLDER
run_id = "%%RUN_ID%%"  # PLACEHOLDER
automl_settings = {'path':None,'name':'capstone-automl-exp','subscription_id':'d7f39349-a66b-446e-aba6-0053c2cf1c11','resource_group':'aml-quickstarts-139371','workspace_name':'quick-starts-ws-139371','region':'southcentralus','compute_target':'cluster-automl2','spark_service':None,'azure_service':'remote','many_models':False,'pipeline_fetch_max_batch_size':1,'iterations':1000,'primary_metric':'AUC_weighted','task_type':'classification','data_script':None,'validation_size':0.0,'n_cross_validations':None,'y_min':None,'y_max':None,'num_classes':None,'featurization':'auto','_ignore_package_version_incompatibilities':False,'is_timeseries':False,'max_cores_per_iteration':1,'max_concurrent_iterations':5,'iteration_timeout_minutes':None,'mem_in_mb':None,'enforce_time_on_windows':False,'experiment_timeout_minutes':30,'experiment_exit_score':None,'whitelist_models':None,'blacklist_algos':['TensorFlowLinearClassifier','TensorFlowDNN'],'supported_models':['BernoulliNaiveBayes','KNN','RandomForest','ExtremeRandomTrees','MultinomialNaiveBayes','GradientBoosting','DecisionTree','TensorFlowLinearClassifier','XGBoostClassifier','AveragedPerceptronClassifier','SGD','LogisticRegression','TensorFlowDNN','LightGBM','SVM','LinearSVM'],'auto_blacklist':True,'blacklist_samples_reached':False,'exclude_nan_labels':True,'verbosity':20,'_debug_log':'azureml_automl.log','show_warnings':False,'model_explainability':True,'service_url':None,'sdk_url':None,'sdk_packages':None,'enable_onnx_compatible_models':True,'enable_split_onnx_featurizer_estimator_models':False,'vm_type':'STANDARD_DS3_V2','telemetry_verbosity':20,'send_telemetry':True,'enable_dnn':False,'scenario':'SDK-1.13.0','environment_label':None,'force_text_dnn':False,'enable_feature_sweeping':False,'enable_early_stopping':True,'early_stopping_n_iters':10,'metrics':None,'enable_ensembling':True,'enable_stack_ensembling':False,'ensemble_iterations':15,'enable_tf':False,'enable_subsampling':None,'subsample_seed':None,'enable_nimbusml':False,'enable_streaming':False,'force_streaming':False,'track_child_runs':True,'allowed_private_models':[],'label_column_name':'target','weight_column_name':None,'cv_split_column_names':None,'enable_local_managed':False,'_local_managed_run_id':None,'cost_mode':1,'lag_length':0,'metric_operation':'maximize','preprocess':True} # PLACEHOLDER for AutoMLSettings

setup_container = "dcid.AutoML_706510b9-e904-4da3-80f4-3cd9ee175e4e_setup" # PLACEHOLDER for SetupRun Container ID to get pkls from there
featurization_json_path = "featurizer_container.json" # PLACEHOLDER for Relative path feature json

feature_list_json = '''{"featurizers":[{"index":0,"transformers":["SimpleImputer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":1,"transformers":["SimpleImputer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":2,"transformers":["SimpleImputer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":3,"transformers":["SimpleImputer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":4,"transformers":["SimpleImputer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":5,"transformers":["CatImputer","StringCastTransformer","LabelEncoderTransformer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":6,"transformers":["StringCastTransformer","CountVectorizer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":7,"transformers":["CatImputer","StringCastTransformer","LabelEncoderTransformer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":8,"transformers":["StringCastTransformer","CountVectorizer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":9,"transformers":["CatImputer","StringCastTransformer","LabelEncoderTransformer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":10,"transformers":["StringCastTransformer","CountVectorizer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":11,"transformers":["StringCastTransformer","CountVectorizer"],"is_distributable":false,"is_separable":false,"is_cached":false},{"index":12,"transformers":["StringCastTransformer","CountVectorizer"],"is_distributable":false,"is_separable":false,"is_cached":false}]}'''

if enable_streaming is not None:
    print("Set enable_streaming flag to", enable_streaming)
    # use_incremental_learning flag will be deprecated and enable_streaming will be used
    automl_settings['use_incremental_learning']=enable_streaming
    automl_settings['enable_streaming']=enable_streaming

def featurization_run():
    global script_directory
    featurization_wrapper(
        script_directory=script_directory,
        dataprep_json=dataprep_json,
        entry_point=entry_point,
        automl_settings=automl_settings,
        task_type=task_type,
        preprocess=preprocess,
        enable_subsampling=enable_subsampling,
        num_iterations=num_iterations,
        setup_container_id = setup_container,
        featurization_json_path = featurization_json_path,
        featurization_json = feature_list_json
    )
    return "Featurization collector run finished"

if __name__ == '__main__':
    try:
        result = featurization_run()
    except Exception as e:
        errors = {'errors': {'exception': e,
                           'traceback': traceback.format_exc()}}
        try:
            current_run = Run.get_submitted_run()
            current_run.add_properties(errors)
        except Exception:
            pass
        raise
    print(result)
