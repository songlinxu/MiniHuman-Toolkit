


class Node_Agent():
    def __init__(self,llm_response_class):
        self.llm_response_class = llm_response_class


    def run_respond(conversation_dict,print_progress,save_log_txt_path=None):
        sys_prompt = 'You are an intelligent assistant who is good at mimicking human discussion in a multi-user collaboration.\n\n'
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

    def step_forward():


    def _store_log(self,input_string,log_file,color=None, attrs=None, print=False):
        with open(log_file, 'a+') as f:
            f.write(input_string + '\n')
            f.flush()
        if(print):
            cprint(input_string, color=color, attrs=attrs)


class Multi_Agent_Brainstorm():
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
        print('\n\nNote that the instruction is very important for you to format the output of LLMs.')
        print('\n\nFor example, you can provide examples to format the exact output of LLMs so that it is easy for you to extract data from LLM answers automatically.')
        print('\n\nHere are example usage for your reference.\n\n')
        print('LLM_Respond_Class_Item = LLM_Respond_GPT_Class(openai_api_key='',model_name="gpt-3.5-turbo-0125",timeout=120,temperature=0,max_tokens=3000)')
        print('Survey_Pipeline_Demo = Multi_Agent_Brainstorm(llm_response_class = LLM_Respond_Class_Item)')
        print('Survey_Pipeline_Demo.init_agent(agent_num = 3, round_num = 3)')
        print('Survey_Pipeline_Demo.run_brainstorm(topic = \'your goal\',instruction = \'your output format\',conversation_json = \'conversation.json\',print_progress=True,save_log_txt_path=\'log.txt\')')

    def _store_log(self,input_string,log_file,color=None, attrs=None, print=False):
        with open(log_file, 'a+') as f:
            f.write(input_string + '\n')
            f.flush()
        if(print):
            cprint(input_string, color=color, attrs=attrs)
    
    def init_agent(self,agent_num,round_num):
        self.agent_num = agent_num
        self.round_num = round_num
        self.agent_dict = {f'agent_{int(i)}': Node_Agent(self.llm_response_class) for i in range(self.agent_num)}

    def run_brainstorm(self,topic,instruction,conversation_json,print_progress=False,save_log_txt_path=None):
        conversation_dict = {'topic':topic,'instruction':instruction}
        
        for round_id in range(self.round_num):
            conversation_dict[f'round_{int(round_id)}'] = {}
            for agent_id in range(self.agent_num):
                agent_response = self.agent_dict[f'agent_{int(agent_id)}'].run_respond(conversation_dict,print_progress,save_log_txt_path)
                conversation_dict[f'round_{int(round_id)}'][f'agent_{int(agent_id)}'] = agent_response

        with open(conversation_json, 'w') as f:
            json.dump(conversation_dict, f, indent=4)
        print('\nConversation_dict has been stored/saved in your local device.')


