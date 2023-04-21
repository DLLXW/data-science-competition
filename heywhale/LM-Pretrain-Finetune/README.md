## GPT-2/Bart/CPT的预训练和微调全流程训练
[数据集地址](https://www.heywhale.com/org/gaiic2023/competition/area/63fef766b4422ee27402289d/leaderboard)
数据来自于一个脱敏的比赛数据集：脱敏后的影像描述与对应影像报告。我们希望通过影像描述来自动生成影响报告。
|列名|	类型	|示例|
|  ----  | ---- | ----  |
|report_ID	|int	|1|
|description|	脱敏后的影像描述，以字为单位使用空格分割|	101 47 12 66 74 90 0 411 234 79 175|
|diagnosis	|脱敏后的诊断报告，以字为单位使用空格分割	|122 83 65 74 2 232 18 44 95|
## 1.Bart/CPT Pretrain
**基于DeepSpeed-Megatron预训练Bart和CPT模型**
### 1.1 数据相关
预训练数据遵循megatron的标准处理范式，但是因为这里是已经脱敏的数据，没法用现成的分词器进行分词转id。所以这里直接跳过分词这一步。
```bash
cd pretrain
#为了符合megatron标准格式，先将原始数据转jsonl格式
python convert_to_json.py
#
python tools/preprocess_data.py \ 
       --input diag_train.json \ #处理好的json格式的数据
       --output-prefix desc_diag \ #输出文件的前缀（随便写）
       --vocab vocab/vocab.txt \ #词表，因为我们无需分，所以代码中不会实际用到，这里随便写（我这里选择直接使用Bert的词表,去Huggingface下载）
       --dataset-impl mmap \ 
       --tokenizer-type Huggingface \
       --split-sentences
```
不需要分词所以对./tools/preprocess_data.py里面139-141行进行了更改，因为数据集中最小id是9，最大1300，这里选择对所有id+100，这么做的目的
#在于预训练过程中利用了Bert的现成vocab.txt，而Bert的词表里面[CLS][SEP][MASK]分别对应的id为101,102,103。
#对数据集+100刚好跳过。当然其实可以不做，自己在预训练的配置文件里面指定这几个特殊符号也行。

### 1.2预训练相关

配置文件（config.json）以及词表（vocab.txt）下载：
- [fnlp/bart-large-chinese](https://huggingface.co/fnlp/bart-large-chinese)放到vocab-bart-large/
- [fnlp/cpt-large](https://huggingface.co/fnlp/cpt-large)放到vocab-cpt-large/

**⚠️注意：** 这里解释一下词表的作用，虽然我们不需要分词，**但是为了更少的改动代码**，选择将词表放到下面。也可以选择不要词表，那么需要自个去改一改./megatron/data/bart_dataset.py下面的各种特殊符号。（另外：除了那些特殊符号，[CLS][SEP][MASK]，其余的词可以删掉，可以减小一点模型的参数量和计算量）
上面👆俩都针对large模型的配置。
```bash
#预训练Bart模型
./run_pretrain_bart.sh
#预训练CPT模型
./run_pretrain_cpt.sh
```
### 1.3 源码细节

以Bart模型为例：
####  **./megatron/model/bart_model.py ** 
改的主要是__init__(self): 注释掉控制模型大小（base/large）的参数，一切以config.json为准。同时采取增量预训练，也即为读取官方预训练好的权重用本数据集增量预训练。若改了词表大小，则记得pop掉模型权重里面那些不匹配的层。

#### **./megatron/data/bart_dataset.py**

（1）58行：标点符号分割符判定，因为本数据集无法确定标点符号的id，（其实也可以看出来哪些是标点符号），所以这里干脆置空。这样会导致的影响是在进行去噪自编码（Denoising Autoencoder，DAE）预训练的时候会少一种句子重排的破坏规则。

（2）282-303行：word_starts(self, source)方法。作者也刚入门LM，根据对源码的阅读，没理解错的话该方法主要是为了确定词组的位置，然后根据这些位置进行DAE的各种破坏原文本的操作。因为本数据集经过了脱敏，所以无法准确确定这些位置（如果没脱敏，源码里面就可以用jieba分词确定了），所以这里我采取了一个最简单的方案是，随机的产生一些“word start position”。

### 1.4预训练好的模型如何使用
```python
weight=torch.load('./checkpoints/bart-large/iter_0004000/mp_rank_00/model_optim_rng.pt')
torch.save(weight['model']['language_model'],'./custom_pretrain_bart/pytorch_model.bin')
#config.json也要放入custom_pretrain_bart/
from transformers import BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained("custom_pretrain_bart/")
```

### 1.5预训练环境
值得一提的是需要apex库，所以需要
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
其余的包参考requirements.txt

### 2.GPT-2的预训练
因为最近LLM这么火，为了更好的理解GPT的原理，这里将GPT-2的最精简的代码拿来盘一盘。代码过于简单，只有两三个文件，建议好好阅读源码。代码见/GPT-2，一看就懂。不再赘述。

**简单说几点：**
- 将描述和诊断数据拼接起来，中间分隔符隔开，记得加上结束符号，这就是一条样本。
- 序列长度设置为256，不够的进行padding
- 输入x进行位移操作x[1:]就是这条样本对应的label
- CausalSelfAttention执行next word预测

对小白的⚠️建议：把GPT-2/model.py里面的实现逐行精读

## 3.finetune
train pipeline整体代码很简单

核心在于构造：
- input_ids
- attention_mask
- decoder_input_ids
- labels

开始符号，结束符号，Pad等符号很重要，不要乱填。详见dataset.py
```python
class DiagDataset(Dataset):
    def __init__(self,df,max_length=128,finetune=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.max_length = max_length
        self.finetune = finetune
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index: int):
        #下面的代码是finetune数据准备的核心
        sample = self.df.iloc[index]
        desc=sample['description']
        diag=sample['diagnosis']
        desc=[101]+[int(i)+100 for i in desc.split(' ')]+[102]
        diag=[101]+[int(i)+100 for i in diag.split(' ')]+[102]
        context_len=len(desc)
        desc=desc+[0]*(self.max_length-len(desc))
        diag=diag+[0]*(self.max_length-len(diag))
        #input_id
        desc_id=np.array(desc)
        #attention mask
        desc_mask=np.array([1]*context_len+[0]*(self.max_length-context_len))
        #
        diag=np.array(diag)
        diag_id=diag[:-1].copy()
        diag_label=diag[1:].copy()
        diag_label[diag[1:]==0]=-100
        return torch.from_numpy(desc_id),torch.from_numpy(desc_mask),torch.from_numpy(diag_id),torch.from_numpy(diag_label)
```
执行训练和推理如下：
```bash
python train.py #
python infer.py #
```
## 总结
该repo以一个比赛脱敏数据集为例，详细介绍了如何针对Bart/CPT/GPT等模型进行完整的pretrain-fintune-inference流程，可以作为入门LLM的很好实操项目。
作者本人也是最近一个来月才开始学习预训练模型相关知识，上面的流程也可以看作学习过程记录，难免有疏漏之处，敬请谅解！


## 参考
[fastnlp/CPT](https://github.com/fastnlp/CPT)

[nanoGPT](https://github.com/karpathy/nanoGPT)