from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
feed_embedding=pd.read_csv('data/wechat_algo_data1/feed_embeddings.csv')
feed_embedd=[]
cnt=0
for i in feed_embedding['feed_embedding'].values:
    cnt+=1
    feed_embedd.append([float(ii) for ii in i.split(' ') if ii!=''])
#
model_pca=PCA(n_components=64)
feed_embedd=model_pca.fit_transform(np.array(feed_embedd))
feed_embedding=pd.concat((feed_embedding,pd.DataFrame(feed_embedd)),axis=1)
feed_embedding.drop(['feed_embedding'],axis=1,inplace=True)
feed_embedding.columns=['feedid']+['feed_embed_'+str(i) for i in range(64)]
feed_embedding.to_csv('data/feed_embeddings_PCA.csv',index=False)