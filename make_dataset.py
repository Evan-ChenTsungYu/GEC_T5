from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import random
import torch

def tokenizer_token(word_list, error_tag, error_type, tokenizer, max_len):
    type_list = ['None', 'R:PREP', 'R:VERB:TENSE', 'R:NOUN', 'R:OTHER', 'R:MORPH', 'R:VERB', 'U:ADV', 'M:PUNCT', 'M:VERB', 'R:WO', 'M:PREP', 'M:DET', 'R:VERB:FORM', 'U:PREP', 'M:PRON', 'M:VERB:TENSE', 'R:NOUN:NUM', 'U:DET', 'R:ORTH', 'UNK', 'M:CONJ', 'U:VERB:TENSE', 'U:PRON', 'R:ADV', 'R:SPELL', 'M:NOUN', 'U:NOUN', 'U:PUNCT', 'R:DET', 'R:VERB:SVA', 'R:PUNCT', 'M:NOUN:POSS', 'U:VERB', 'U:PART', 'R:CONTR', 'U:OTHER', 'M:VERB:FORM', 'R:ADJ', 'R:ADJ:FORM', 'M:OTHER', 'M:PART', 'M:ADV', 'R:PRON', 'M:CONTR', 'U:CONTR', 'R:PART', 'M:ADJ', 'U:CONJ', 'R:NOUN:INFL', 'U:VERB:FORM', 'R:NOUN:POSS', 'R:VERB:INFL', 'R:CONJ', 'U:ADJ', 'U:NOUN:POSS']

    type_to_class = {k:v  for v, k in enumerate(type_list)}
    class_to_type = {str(v):k  for v, k in enumerate(type_list)}  

    token_text = tokenizer(word_list, is_split_into_words=True)
    
    token_error_tag = []
    token_error_type = []
    for i_token in range(0, len(word_list)):
        start, end = token_text.word_to_tokens(i_token)
        # print(start, end)
        for i_word in range(start, end):
            token_error_tag.append(error_tag[i_token])

            if error_type[i_token] not in type_list: # This is used for conll14 test set who didn't use ERRANT as scorer
                token_error_type.append(type_to_class['None']) 
            else:
                token_error_type.append(type_to_class[error_type[i_token]])
    if len(token_error_tag) >  max_len:
        return token_error_tag[:max_len], token_error_type[:max_len]
    else:
        for i in range(len(token_error_tag), max_len):
            token_error_tag.append(0)
            token_error_type.append(type_to_class['None'])
        return token_error_tag[:max_len], token_error_type[:max_len]
         
def make_error_tag(input_files):
    words = []
    head_ptr = 0
    start_id = 0
    end_id = 0
    output = []
    
    # for file_name in list_files:
    with open(input_files+ '.m2', 'r') as input_file:
        prompt_sent = []
        correct_sent = []
        error_tags = []
        error_types = []
        for line in input_file:
            line = line.strip()
            if line.startswith('S') and head_ptr == 0:
                data_frame = {'origin':[], 'corrected':[],'prompt':[], 'error_tag' : [], 'error_type' : []}
                line = line[2:]
                words = line.split()
                for i in range(0, len(words)+1):
                    error_tags.append(0)
                    error_types.append('None')
            elif line.startswith('A'):
                line = line[2:]
                info = line.split("|||")
                start_id, end_id = info[0].split()
                start_id = int(start_id)
                end_id = int(end_id)
                error_type = info[1]
                annotator = int(info[5])
                error_modified = info[2]
                if annotator == 0:
                    if start_id == -1:
                        correct_sent = words
                        prompt_sent = words
                        head_ptr =  len(words)
                    else:
                        prompt_sent += words[head_ptr:start_id]
                        correct_sent += words[head_ptr:start_id]
                        correct_sent += [error_modified]
                        if start_id == end_id:
                            prompt_sent += ([ '[ ' ] + ['NONE'] + [ '|' , error_modified ,']'])
                            # print(start_id, len(error_tags), words, len(error_tags))
                            error_tags[start_id] = 1
                            error_types[start_id] = error_type
                        else:
                            prompt_sent += ([ '[ ' ] + words[start_id:end_id] + [ '|' , error_modified ,']'])
                            for i in range(start_id, end_id):
                                error_tags[i] = 1
                                error_types[start_id] = error_type       
                        head_ptr =  end_id 
                else:
                    # print(annotator)
                    continue
            elif line.startswith('S') :
                line = line[2:]
                correct_sent += words[head_ptr:]
                prompt_sent += words[head_ptr:]
                data_frame['corrected'] = correct_sent 
                data_frame['prompt'] = prompt_sent 
                data_frame['origin'] = words
                data_frame['error_tag'] = error_tags
                data_frame['error_type'] = error_types
                output.append(data_frame)

                data_frame = {'origin':[], 'corrected':[],'prompt':[], 'error_tag' : [], 'error_type' : []}
                prompt_sent = []
                correct_sent = []
                error_tags = []
                error_types = []
                head_ptr = 0
                words = line.split()
                for i in range(0, len(words)+1):
                    error_tags.append(0)
                    error_types.append('None')

        correct_sent += words[head_ptr:]
        data_frame['prompt'] = prompt_sent
        data_frame['corrected'] = correct_sent
        data_frame['origin'] = words
        data_frame['error_tag'], data_frame['error_type'] = error_tags, error_types
        output.append(data_frame)

        return output



def make_label(dataset_name, type):
    path = '/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/m2_' + type+'/'
    if dataset_name != 'wi':
        file_name = dataset_name + '.' + type 
        data = make_error_tag(path+file_name)
        return data
    else:
        if type == 'train':
            wi_list = ['A', 'B', 'C', 'ABC']
        else:
            wi_list = ['A', 'B', 'C', 'ABCN', 'N'] 
        data = []
        for i in wi_list:
            file_name = dataset_name + '.'+ i + '.' + type
            data += make_error_tag(path+file_name)
        return data

class GEC_DS(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        
        wrong = self.data[index]['origin']
        corrected = self.data[index]['corrected']
        prompt = self.data[index]['prompt']
        error_tag = self.data[index]['error_tag']
        error_type = self.data[index]['error_type']
        
        input_data = self.tokenizer(wrong, is_split_into_words=True, add_special_tokens = True, padding='max_length', max_length = self.max_len, truncation=True, return_tensors = 'pt')
        output_text_label = self.tokenizer(corrected, is_split_into_words=True, add_special_tokens = True, padding='max_length', max_length = self.max_len, truncation=True, return_tensors = 'pt')
        prompt_data =  self.tokenizer(prompt, is_split_into_words=True, add_special_tokens = True, padding='max_length', max_length = self.max_len, truncation=True, return_tensors = 'pt')
        token_error_tag, token_error_type = tokenizer_token(wrong, error_tag, error_type, self.tokenizer, max_len = self.max_len)
        
        # print(token_error_tag, token_error_type)
        return {'Data':input_data['input_ids'],'Data_Mask':input_data['attention_mask'], 'text_label': output_text_label['input_ids'], 'text_mask': output_text_label['attention_mask'],
        'prompt_label' :prompt_data['input_ids'], 'prompt_mask': prompt_data['attention_mask'], 'error_tag' :torch.IntTensor(token_error_tag), 'error_type' : torch.IntTensor(token_error_type)}
       

def save_dataset(ds_name, data_type, tokenizer, max_len):
    data = make_label(ds_name, data_type) 
    dataset = GEC_DS(data, tokenizer, max_len)
    torch.save(dataset, '/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/' +data_type + '/' + ds_name + '.pt')

def load_dataset(ds_name, type):
    return torch.load('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/' + type + '/' + ds_name + '.pt')

def load_GEC_dataset(dataset_dict, TT, batch_size, split_size = 1, val_split_size = 0.01): #dataset list should contain [(ds_name, type), ]


    TT_ds_list = []
    Val_ds_list = []
    for  i_ds in range(0, len(dataset_dict[TT])):
         TT_ds_list.append(load_dataset(dataset_dict[TT][i_ds], TT))
    
    for  i_ds in range(0, len(dataset_dict['valid'])):
         Val_ds_list.append(load_dataset(dataset_dict['valid'][i_ds], 'valid'))
    
    TT_DS = ConcatDataset(TT_ds_list)
    Val_DS = ConcatDataset(Val_ds_list)

    if TT != 'test':
        tt_index =  random.sample(range(0, len(TT_DS)), int(split_size*len(TT_DS)))
        TT_DS = Subset(TT_DS, indices= tt_index)
    
    val_index = random.sample(range(0, len(Val_DS)), int(val_split_size*len(Val_DS)))
    Val_DS = Subset(Val_DS, indices= val_index)
    
    if TT != 'test':
        TT_DL = DataLoader(TT_DS, batch_size = batch_size, shuffle=True)
    else:
        TT_DL = DataLoader(TT_DS, batch_size = batch_size)
    Val_DL = DataLoader(Val_DS, batch_size = batch_size)

    return TT_DL, Val_DL

if __name__ == '__main__':
    from transformers import BertTokenizerFast
    from datasets import load_dataset
    from make_label import classify_label, text_label,make_label
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer

    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    tokenizer.add_tokens(['|', '[', ']', 'NONE'])
    # print(len(tokenizer))
    # save_dataset('wi', 'train', tokenizer = tokenizer, max_len = 50)
    test_load_data = torch.load('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/train/wi.pt')
    # # print(test_load_data[2])
    # # print(test_load_data[2]['Data'])
    # print(tokenizer.decode(test_load_data[2]['Data'][0,:],skip_special_tokens = True))
    # print(tokenizer.decode(test_load_data[2]['text_label'][0,:],skip_special_tokens = True))
    # print(tokenizer.decode(test_load_data[2]['prompt_label'][0,:],skip_special_tokens = True))


    # for i_type in ['valid', 'train']:
    #     try:
    #         save_dataset('wi', i_type, tokenizer = tokenizer, max_len = 50)
    #         save_dataset('fce', i_type, tokenizer = tokenizer, max_len = 50)
    #         save_dataset('lang8', i_type, tokenizer = tokenizer, max_len = 50)
    #     except:
    #         continue
    
    test_load_data = torch.load('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/train/lang8.pt')
    # for i in range(0, len(test_load_data)):
    #     # print(i, test_load_data[i])
    #     continue