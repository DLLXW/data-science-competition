import os
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
last90=open('subqyl_722.txt','r')
sub0=last90.readlines()
merge_sub=open('subqyl_726.txt','w')
subs={}
for model_name in ['mobilenet_v2']:
    for fold in range(1,5+1):
        read_sub=open('submitKfold/'+'submit_'+model_name+'Kfold'+str(fold)+'.txt','r')
        subs[model_name+'Kfold'+str(fold)]=read_sub.readlines()
re_dic={'alpha':0,'beta':0,'betax':0}
sub_most=[]
#
for i in range(len(sub0)):
    index=sub0[i].split(' ')[0]
    tmp={'1':0,'2':0,'3':0}
    merges=[value[i].strip('\n').split(' ')[1] for _,value in subs.items()]
    for k in merges:
        tmp[k]+=1
    #print(tmp)
    most=sorted(tmp.items(),key=lambda item:item[1])[-1][0]
    #print(most)
    sub_most.append(most)
    merge_sub.write(index+' '+most)
    if i!=1171:
        merge_sub.write('\n')
    if most=='1':
        re_dic['alpha']+=1
    elif most=='2':
        re_dic['beta']+=1
    else:
        re_dic['betax']+=1
print(re_dic)
#
sub0=[int(p.strip('\n').split(' ')[1]) for p in sub0]
sub_most=[int(p) for p in sub_most]
_,_,f_class,_=precision_recall_fscore_support(sub_most,sub0)
fper_class = {'beta': f_class[1], 'betax': f_class[2], 'alpha': f_class[0]}
print('各类单独F1:', fper_class, '各类F1取平均：', f_class.mean())
for key,value in subs.items():
    sub=[int(p.strip('\n').split(' ')[1]) for p in value]
    print('{}/sub90:{}'.format(key,accuracy_score(sub,sub0)))
    print('{}/sub_most:{}'.format(key, accuracy_score(sub, sub_most)))