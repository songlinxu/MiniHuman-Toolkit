import logging
# from transformers import logging as transformers_logging
# transformers_logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, EncoderDecoderModel, ReformerModelWithLMHead, ReformerTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
import os,sys,random,time,shutil,warnings

from sklearn.metrics import accuracy_score, f1_score

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)  # Use shutil.rmtree to remove the directory and its contents
        print(f"Folder '{folder_path}' and all its contents have been removed successfully.")
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the folder: {e}")



class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_prompt = self.dataframe.iloc[idx]['input_prompt']
        # input_prompt_1 = self.dataframe.iloc[idx]['input_past_prompt']
        # input_prompt_2 = self.dataframe.iloc[idx]['input_future_prompt']
        # input_prompt_3 = self.dataframe.iloc[idx]['initial_response']
        # input_prompt_4 = self.dataframe.iloc[idx]['best_reflection']
        label = self.dataframe.iloc[idx]['future_label']
        
        # inputs = self.tokenizer(input_prompt_2, input_prompt_3, input_prompt_4, truncation='longest_first', padding='max_length', max_length=self.max_length, return_tensors='pt')
        # inputs = self.tokenizer(input_prompt_3, truncation='longest_first', padding='max_length', max_length=self.max_length, return_tensors='pt')
        inputs = self.tokenizer(input_prompt_1, input_prompt_2, truncation='longest_first', padding='max_length', max_length=self.max_length, return_tensors='pt')

        item = {key: val.squeeze() for key, val in inputs.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        # print('input_token_len: ',len(self.tokenizer.tokenize(input_prompt_1))+len(self.tokenizer.tokenize(input_prompt_2)))

        return item


class lm_classifier_pipeline():
    def __init__(self, max_length, log_folder, load_model_type, random_seed = 4, lr=1e-4):
        self.max_length = max_length
        self.set_seed(random_seed)

        self.log_folder = log_folder
        self.load_model_type = load_model_type

        self.model_init(load_model_type,lr)

    def set_seed(self,seed_num):
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    def dataset_prepare(self,dataset_train_path,dataset_test_path,batch_size = 16):
        dataframe_train = pd.read_csv(dataset_train_path,sep='\t')
        dataframe_test = pd.read_csv(dataset_test_path,sep='\t')
        
        dataset = CustomDataset(dataframe_train, self.tokenizer, self.max_length)
        test_dataset = CustomDataset(dataframe_test, self.tokenizer, self.max_length)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size 
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    def model_save(self,checkpoint,checkpoint_path,tokenizer,tokenizer_path,e2emodel,e2emodel_path):
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
        torch.save(checkpoint, checkpoint_path)
        if os.path.exists(tokenizer_path): remove_folder(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
        if os.path.exists(e2emodel_path): remove_folder(e2emodel_path)
        e2emodel.save_pretrained(e2emodel_path)


    def model_init(self,load_model_type,lr=1e-4):
        assert load_model_type in ['best','last','none']

        self.checkpoint_last_path = self.log_folder + '/model_last.pt'
        self.checkpoint_best_path = self.log_folder + '/model_best.pt'
        self.tokenizer_last_path = self.log_folder + '/tokenizer_last'
        self.tokenizer_best_path = self.log_folder + '/tokenizer_best'
        self.e2emodel_last_path = self.log_folder + '/e2emodel_last'
        self.e2emodel_best_path = self.log_folder + '/e2emodel_best'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        if load_model_type in ['best','last']:
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_last_path) if load_model_type == 'last' else BertTokenizer.from_pretrained(self.tokenizer_best_path)
            self.model = BertForSequenceClassification.from_pretrained(self.e2emodel_last_path).to(self.device) if load_model_type == 'last' else BertForSequenceClassification.from_pretrained(self.e2emodel_best_path).to(self.device)
            checkpoint = torch.load(self.checkpoint_last_path) if load_model_type == 'last' else torch.load(self.checkpoint_best_path)
            self.epoch_exist = checkpoint['epoch']
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_loss']
            self.val_losses = checkpoint['val_loss']
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.epoch_exist = 0
            self.train_losses = []
            self.val_losses = []

        # self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def model_train(self, epochs=3, vis=True):
        loss_file = self.log_folder+'/loss.csv'
        with open(loss_file, "a+") as file1:
            file1.write('epoch,train_loss,val_loss,accuracy,f1\n')

        for epoch in range(epochs):
            if epoch + 1 <= self.epoch_exist: continue
            self.model.train()
            total_loss = 0
            time_train_start = time.time()
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # outputs = self.model(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     labels=labels
                # )

                # print('outputs: ',outputs)

                # loss = outputs.loss

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.logits, labels)
                # loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            time_train_end = time.time()

            avg_loss = total_loss / len(self.train_dataloader)
            self.train_losses.append(avg_loss)
            train_time = time_train_end - time_train_start
            print(f'Epoch {epoch + 1}, Training loss: {avg_loss}, Training Time: {train_time}')
            
            val_loss, accuracy, f1 = self.model_eval(eval_mode='val')

            with open(loss_file, "a+") as file1:
                file1.write(str(epoch + 1)+','+str(avg_loss)+','+str(val_loss)+','+str(accuracy)+','+str(f1)+'\n')

            checkpoint = {
                'epoch': epoch + 1,
                # 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses,
                'val_loss': self.val_losses
            }
            
            self.model_save(checkpoint,self.checkpoint_last_path,self.tokenizer,self.tokenizer_last_path,self.model,self.e2emodel_last_path)

            if (len(self.val_losses) == 0) or (len(self.val_losses) != 0 and val_loss == min(self.val_losses)):
                self.model_save(checkpoint,self.checkpoint_best_path,self.tokenizer,self.tokenizer_best_path,self.model,self.e2emodel_best_path)

        if vis == True:
            fig, ax = plt.subplots()
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.savefig(self.log_folder+'/loss.png')

    def model_eval(self,eval_mode):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        assert eval_mode in ['test','val']
        dataloader_part = self.test_dataloader if eval_mode == 'test' else self.val_dataloader
        time_eval_start = time.time()

        with torch.no_grad():
            for batch in dataloader_part:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)

                outputs = self.model(input_ids, attention_mask)
                # outputs = self.model(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     labels=labels
                # )
                # loss = outputs.loss
                loss = self.criterion(outputs.logits, labels)
                # loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        time_eval_end = time.time()
        time_eval = time_eval_end - time_eval_start
        avg_loss = total_loss / len(dataloader_part)
        self.val_losses.append(avg_loss)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'{eval_mode} loss: {avg_loss}, accuracy: {accuracy}, F1 score: {f1}, time: {time_eval}')
        return avg_loss, accuracy, f1



