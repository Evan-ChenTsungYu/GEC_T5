
import json
import pandas as pd
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
                error_modified = info[2]
                
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



if __name__ == '__main__':
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    test_label = make_label('lang8','train')
    for i in range(0,20):
        print(test_label[i])
    # print(len(test_label['wrong']), len(test_label['wrong']), len(test_label['error_tag']),len(test_label['error_type']))

    
   
    
    