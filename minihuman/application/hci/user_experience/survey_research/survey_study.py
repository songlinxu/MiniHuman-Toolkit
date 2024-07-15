import json,os
import numpy as np 
import pandas as pd
from termcolor import colored, cprint
from .llm_api import LLM_Respond_GPT_Class
from .demo import demo_survey, demo_user_persona

class Survey_Pipeline():
    def __init__(self,llm_response_class = None,openai_api_key = None,llm_model_name="gpt-3.5-turbo-0125",timeout=120,temperature=0,max_tokens=3000):
        '''
        The llm_response_class should at least have a run_response function to accept message list as input.
        Use .help() function to get more info about how to use this class.
        '''
        if llm_response_class == None:
            print('You did not set your own llm_response_class. We will use the default openai GPT 3.5 model. Please at least set your openai api_key.')
            self.llm_response_class = LLM_Respond_GPT_Class(openai_api_key=openai_api_key,model_name=llm_model_name,timeout=timeout,temperature=temperature,max_tokens=max_tokens)
        else:
            self.llm_response_class = llm_response_class
            print('Load your LLM successfully!')
        
    def help(self):
        self.load_default_user_persona()
        self.load_default_survey()
        print('\n\nNote that the instruction item in each question is very important for you to format the output of LLMs.')
        print('\n\nFor example, you can provide examples to format the exact output of LLMs so that it is easy for you to extract data from LLM answers automatically.')
        print('\n\nHere are example usage for your reference.\n\n')
        print('LLM_Respond_Class_Item = LLM_Respond_GPT_Class(openai_api_key='',model_name="gpt-3.5-turbo-0125",timeout=120,temperature=0,max_tokens=3000)')
        print('Survey_Pipeline_Demo = Survey_Pipeline(llm_response_class = LLM_Respond_Class_Item)')
        print('Survey_Pipeline_Demo.init_user(user_json_file = \'user_demo.json\')')
        print('Survey_Pipeline_Demo.init_survey(survey_json_file = \'survey_demo.json\')')
        print('Survey_Pipeline_Demo.run_survey(output_csv_path = \'survey_data_demo.csv\',repeat_num=2,save_log_txt_path=\'log.txt\',print_progress=True)')

    def load_json_file(self,file_name):
        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)
        # Construct the path to the JSON file
        json_file_path = os.path.join(current_dir, file_name)
        
        # Open and read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        return data

    def _store_log(self,input_string,log_file,color=None, attrs=None, print=False):
        with open(log_file, 'a+') as f:
            f.write(input_string + '\n')
            f.flush()
        if(print):
            cprint(input_string, color=color, attrs=attrs)

    def load_default_user_persona(self):
        user_persona_dict = demo_user_persona
        # user_persona_dict = self.load_json_file('user_demo.json')
        # with open('./user_demo.json', 'r') as f:
        #     user_persona_dict = json.load(f)
        print(f'Here is the default example user personas.\n\n{user_persona_dict}')
        return user_persona_dict

    def load_default_survey(self):
        survey_dict = demo_survey
        # survey_dict = self.load_json_file('survey_demo.json')
        # with open('./survey_demo.json', 'r') as f:
        #     survey_dict = json.load(f)
        print(f'Here is the default example survey.\n\n{survey_dict}')
        return survey_dict

    def generate_default_user_persona(self):
        user_persona_dict = {
            'u1': {'age': '24 years old', 'job': 'engineer'},
            'u2': {'education': 'high school'},
            'u3': {'name': 'Tom', 'major': 'Computer Science', 'degree': 'Master'},
        }
        print(f'Here is the default example user personas.\n\n{user_persona_dict}')
        with open('user_demo.json', 'w') as f:
            json.dump(user_persona_dict, f, indent=4)
        print('\nDemo user has been stored/saved in your local package.')

    def generate_default_survey(self):
        survey_dict = {
            'q1': {'type': 'text-entry', 'content': 'What is your name?', 'action': 'text entry', 'instruction': 'Output your answers with reasons within 20 words.'},
            'q2': {'type': 'single-choice', 'content': 'What is your favorite fruit?', 'action': 'A. apple, B. banana, C. lemon', 'instruction': 'Output your single choice directly without reasons. Your output format should exactly like: choice: [choice].'},
            'q3': {'type': 'multiple-choice', 'content': 'What do you eat everyday?', 'action': 'A. noodles, B. rice, C. meat', 'instruction': 'Output your multiple choices directly with reasons. Your output format should be exactly like: choice: [choice], reason: []. For example, if you want to select A and C, your output should be: choice: A,C, reason: [my reasons...].'},
        }
        print(f'Here is the default example survey.\n\n{survey_dict}')
        with open('survey_demo.json', 'w') as f:
            json.dump(survey_dict, f, indent=4)
        print('\nDemo survey has been stored/saved in your local package.')

        
    def init_user(self,user_json_file = None):
        '''
        The user_json file should include [user_id,user_persona]
        '''
        if user_json_file == None:
            self.user_persona_dict = self.load_default_user_persona()
            print('No init users. We will load default demo users.')
        else:
            with open(user_json_file, 'r') as f:
                self.user_persona_dict = json.load(f)
            print('Load your users successfully!')

    def init_survey(self,survey_json_file = None):
        '''
        Your survey format must be a json file including...
        '''
        if survey_json_file == None:
            self.survey_dict = self.load_default_survey()
            print('No init survey. We will load default demo survey.')
        else:
            with open(survey_json_file, 'r') as f:
                self.survey_dict = json.load(f)
            print('Load your survey successfully!')

    def _get_user_persona_str(self,user_persona_dict):
        user_persona_str = '\n\nHere is your persona information: '
        persona_type_list = list(user_persona_dict.keys())
        persona_type_list.sort()
        for persona_type in persona_type_list:
            persona_content = user_persona_dict[persona_type]
            user_persona_str += f'Your #{persona_type}# is {persona_content}. '
        user_persona_str += '\n\n'
        return user_persona_str

    def _get_question_info_str(self,question_info_dict):
        try:
            question_type = question_info_dict['type']
            question_content = question_info_dict['content']
            question_action = question_info_dict['action']
            question_instruction = question_info_dict['instruction']
        except:
            raise ValueError('Your survey json file must have type/content/action/instruction for each question. Please use Survey_Pipeline.help() to find an example of how to use it.')
        if question_type == 'single-choice': 
            question_action_str = f'You need to select only one choice from all choices below: {question_action}'
        elif question_type == 'multiple-choice': 
            question_action_str = f'You can select multiple choices from all choices below: {question_action}'
        else:
            question_action_str = f'The actions you can take for the question is: {question_action}. \n'
        question_info_str = (
            '\n\nHere is question that you need to answer: \n'
            +f'The #question type# is: {question_type}. \n'
            +f'The #question content# is: {question_content}. \n'
            +question_action_str
            +f'To answer the question and format your output, the #question instruction# is below: {question_instruction}. \n\n'
        )
        return question_info_str

    def generate_response(self,user_persona_dict,question_info_dict,save_log_txt_path,print_progress):
        user_persona_str = self._get_user_persona_str(user_persona_dict)
        question_info_str = self._get_question_info_str(question_info_dict)
        sys_prompt = 'You are an intelligent assistant who is good at mimicking humans with specific personas to answer contextual questions in a survey.\n\n'
        input_prompt = user_persona_str+question_info_str
        message_sys = {"role": "system","content": sys_prompt}
        message_user = {"role": "user","content": input_prompt}
        message_list = [message_sys,message_user]
        response_str = self.llm_response_class.run_response(message_list,print_progress)
        if save_log_txt_path != None:
            if print_progress:
                self._store_log('\n\system prompt:'+sys_prompt,save_log_txt_path,color='blue',print=True)
                self._store_log('\n\ninput prompt:'+input_prompt,save_log_txt_path,color='red',print=True)
                self._store_log('\n\nllm response:'+response_str,save_log_txt_path,color='green',print=True)
            else:
                self._store_log('\n\system prompt:'+sys_prompt,save_log_txt_path,print=False)
                self._store_log('\n\ninput prompt:'+input_prompt,save_log_txt_path,print=False)
                self._store_log('\n\nllm response:'+response_str,save_log_txt_path,print=False)
        return response_str


    def run_survey(self,output_csv_path,repeat_num = 1,save_log_txt_path = None,print_progress=False):
        survey_data_header = ['user_id','repeat_id','question_id','response']
        survey_data = []
        user_list = list(self.user_persona_dict.keys())
        user_list.sort()
        question_list = list(self.survey_dict.keys())
        question_list.sort()
        for user_id in user_list:
            user_each_dict = self.user_persona_dict[user_id]
            for r in range(repeat_num):
                for question_id in question_list:
                    question_each_dict = self.survey_dict[question_id]
                    response = self.generate_response(user_each_dict,question_each_dict,save_log_txt_path,print_progress)
                    survey_data.append([user_id,r,question_id,response])
                    if print_progress:
                        print(f'user {user_id} finished question {question_id} in repetition {r}.')

        survey_data = pd.DataFrame(np.array(survey_data),columns=survey_data_header)
        survey_data.to_csv(output_csv_path,index=False)

