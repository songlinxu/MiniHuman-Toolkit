from .lm_classifier import LM_Classifier_Pipeline
import pandas as pd 


def run_exp_bertKT():
    dataset_path = 'dataset.csv'
    log_folder = './test/'

    dataframe = pd.read_csv(dataset_path,sep='\t')
    dataframe_train = dataframe[dataframe['data_type']=='train'].iloc[0:100]
    dataframe_valid = dataframe[dataframe['data_type']=='test'].iloc[0:50]
    dataframe_test = dataframe[dataframe['data_type']=='test'].iloc[0:50]
    
    experiment_pipeline = LM_Classifier_Pipeline(512,log_folder,load_model_type='none',lr=5e-7)
    experiment_pipeline.dataset_prepare(dataframe_train,dataframe_valid,dataframe_test,input_column_list = ['past_question_answer_prompt','future_question_prompt'],label_column = 'correctness',trim_type = 'longest_first',batch_size = 16)
    experiment_pipeline.model_train(epochs=3,vis=True)
    experiment_pipeline.model_eval(eval_mode='test')

    experiment_pipeline = LM_Classifier_Pipeline(512,log_folder,load_model_type='best',lr=5e-7)
    experiment_pipeline.dataset_prepare(dataframe_train,dataframe_valid,dataframe_test,input_column_list = ['past_question_answer_prompt','future_question_prompt'],label_column = 'correctness',trim_type = 'longest_first',batch_size = 16)
    experiment_pipeline.model_eval(eval_mode='test')

    experiment_pipeline = LM_Classifier_Pipeline(512,log_folder,load_model_type='best',lr=5e-7)
    all_preds, accuracy, f1 = experiment_pipeline.model_predict(dataframe_test, input_column_list = ['past_question_answer_prompt','future_question_prompt'], label_column = None, trim_type = 'longest_first')
    print(all_preds)
    print(accuracy, f1)

run_exp_bertKT()