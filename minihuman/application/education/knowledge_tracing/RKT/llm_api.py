import os, sys, time, json, math 
import numpy as np 
import pandas as pd 

import openai
from openai import OpenAI

def _response_llm_gpt(message_list,openai_api_key,model_name="gpt-3.5-turbo-0125",timeout=120,temperature=0,max_tokens=3000):
    openai.api_key = openai_api_key
    client = OpenAI(api_key = openai.api_key)
    response = ''
    except_waiting_time = 1
    max_waiting_time = 32
    current_sleep_time = 0.5
    
    start_time = time.time()
    while response == '':
        try:
            completion = client.chat.completions.create(
                model=model_name, # gpt-4o, gpt-4
                messages=message_list,
                temperature=temperature,
                timeout = timeout,
                max_tokens=max_tokens
            )
                
            response = completion.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(current_sleep_time)
            if except_waiting_time < max_waiting_time:
                except_waiting_time *= 2
            current_sleep_time = np.random.randint(0, except_waiting_time-1)
    end_time = time.time()   
    print('llm response time: ',end_time-start_time)     
    return response

class LLM_Respond_GPT_Class():
    def __init__(self,openai_api_key,model_name="gpt-3.5-turbo-0125",timeout=120,temperature=0,max_tokens=3000):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run_response(self,message_list):
        return _response_llm_gpt(message_list,self.openai_api_key,self.model_name,self.timeout,self.temperature,self.max_tokens)

