import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Cluster_Text_Data():
    def __init__(self):
        pass 

    def help(self):
        help_str = '''
            dataframe = pd.read_csv('fake_data.csv',sep='\t')
            Cluster_Text_Data_Class = Cluster_Text_Data()
            Cluster_Text_Data_Class.load_process_dataset(dataframe,cluster_column='llm_response',stop_words='english')
            clustered_labels = Cluster_Text_Data_Class.run_cluster(n_clusters=5,model_name='kmeans',random_state=0)
            Cluster_Text_Data_Class.visual_cluster(vis_model='pca',savefig_path='temp.png')
            print('clustered_labels: ',clustered_labels)
        '''
        print(help_str)

    def load_process_dataset(self,dataframe,cluster_column,stop_words='english'):
        self.df = dataframe
        self.df['text'] = self.df[cluster_column].str.lower()

        vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.X = vectorizer.fit_transform(self.df['text'])

    def run_cluster(self,n_clusters,model_name='kmeans',random_state=0):
        if model_name == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(self.X)
            self.df['cluster'] = kmeans.labels_
        else:
            raise ValueError('This cluster model is not supported yet!')
            
        return np.array(self.df['cluster'])

    def visual_cluster(self,vis_model='pca',savefig_path='temp.png'):
        if vis_model == 'pca':
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(self.X.toarray())
            df_pca = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])
            df_pca['cluster'] = self.df['cluster']

            plt.figure(figsize=(8,6))
            plt.scatter(df_pca['principal component 1'], df_pca['principal component 2'], c=df_pca['cluster'])
            plt.title('Text Clustering with KMeans')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.savefig(savefig_path)
            plt.show()
        else:
            raise ValueError('This visualization model is not supported yet!')

