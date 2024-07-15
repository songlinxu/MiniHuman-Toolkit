import os,sys,re,time
from datetime import datetime
from TIR import Transferable_Iterative_Reflection

def check_between_subject(dataset_path):
    pass 


def reflect_from_train_set(dataset_path,result_root,openai_api_key,llm_model_type,reflect_student_num,iter_threshold=3,reflect_sample_ratio=1):
    '''
    The dataset_path should be your TRAIN set
    Your input dataset columns MUST be: ['uid','data_type','xy_type','category','student_id','question_id','concept_id','correctness','question_content','concept_content']
    If your datasets do not have some columns, you still need to put None for these columns.
    
    [uid]: the unique identifier for each sample. Each sample is one question from a specific student. You can understand the uid as the unique index for the table. 
    But you MUST use uuid python library to generate a unique uid per row. You can NOT just use index to be uid.

    [data_type]: this value is either 'train' or 'test'. It determines whether each sample will be used for training or testing. 
    Note that the data_type assignment MUST be between-subject for students (same as pyKT). 
    This means all questions from one student should be all 'train' or all 'test'.
    You can NOT make some questions from one student to be 'train' but some other questions from the same student to be 'test'.

    This means you should first divide students into either train or test group based on a split ratio. After that, you should assign all rows from one student to either
    'train' or 'test'.

    [xy_type]: this value is either 'past' or 'future'. 


    [category]:

    [concept_id]: If a question_id is related to multiple concepts, you should expand it into multiple rows based on the multiple concepts. 
    For example, if question 1 is related to concepts 2,3,4, then you should have three rows: [1,2],[1,3],[1,4] for [question_id,concept_id]. 
    
    However, when you are splitting the dataset into train/test set, you MUST pay attention that all samples related to this question should be put into the same set (either train/test).
    You can NOT put some (such as [1,2]) in train set but some (such as [1,3]) in test set. This will result in label leakage because our evaluation is question-level.

    reflect_sample_ratio: the ratio (0~1) of students used for reflection.  
    reflect_student_num: how many students you want to select for reflection. Note that the students will be selected AFTER filtering students with reflect_sample_ratio.
    iter_threshold: how many iterations for reflection. We recommend it to be 3~5.
    '''
    check_between_subject(dataset_path)
    log_folder = result_root+'/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    pipeline = Transferable_Iterative_Reflection(llm_model_type=llm_model_type,dataset_path=dataset_path,log_folder=log_folder,print_log=True,openai_api_key=openai_api_key)

    pipeline.dataset_prepare(datasample_flag=True,reflect_sample_ratio=reflect_sample_ratio,random_seed=4)
    pipeline.run_reflect(student_num=reflect_student_num,iter_threshold=iter_threshold)



def reflect_test_set_from_train_set(dataset_path,result_root,openai_api_key,llm_model_type,example_student_num_per_category,test_student_num_per_category,iter_threshold=3):
    check_between_subject(dataset_path)
    log_folder = result_root+'/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    pipeline = Transferable_Iterative_Reflection(llm_model_type=llm_model_type,dataset_path=dataset_path,log_folder=log_folder,print_log=True,openai_api_key=openai_api_key)

    pipeline.run_predict_reuse_reflect(example_student_num_per_course=example_student_num_per_category,test_student_num_per_course=test_student_num_per_category,random_seed=4,iter_threshold=iter_threshold)




# calculate_metrics('/home/songlin/study_results/GKT/reflection_generation/direct_reflect_reuse_history_all_8past/predict_post_detail.csv')


def run_exp_bertLLMKT(dataset_train_path,dataset_test_path,log_folder):
    experiment_pipeline = Experiment_Pipeline(512,log_folder,load_model_type='none',lr=5e-7)
    experiment_pipeline.dataset_prepare(dataset_train_path,dataset_test_path,batch_size=64)
    experiment_pipeline.model_train(epochs=50,vis=True)
    experiment_pipeline.model_eval(eval_mode='test')

    experiment_pipeline = Experiment_Pipeline(512,log_folder,load_model_type='best',lr=5e-7)
    experiment_pipeline.dataset_prepare(dataset_train_path,dataset_test_path,batch_size=64)
    experiment_pipeline.model_eval(eval_mode='test')
