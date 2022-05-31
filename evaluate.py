import torch.nn as nn
import torch
from sklearn.metrics import fbeta_score, confusion_matrix, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
import errant
import os
import time
from tqdm import tqdm

def F_score(output, label):
    batch_size = output.size()[0]
    F_score = 0
    output = output.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    # print(output.shape, label.shape)
    for i_batch in range(0, batch_size):
        F_score += fbeta_score(output[i_batch, 1:], label[i_batch, :-1], labels = torch.arange(start=1, end = 32128, step=1), average='micro', beta=0.5)
    return F_score/batch_size

def evaluate(Model, TT_DL, max_len, DEVICE, tokenizer, joint_training):
    text_fscore = 0
    class_fscore = 0
    file_path = './result/' + joint_training + '/'
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)
    print('Delete Existed file')
    file_list = [ f for f in os.listdir(file_path) ]
    for f in file_list:
        os.remove(os.path.join(file_path, f))
    for index, content in enumerate(tqdm(TT_DL)):
        correct_text_label = content['text_label'].squeeze(1).to(DEVICE)
        correct_text_mask = content['text_mask'].squeeze(1).to(DEVICE)
        wrong_text_label = content['Data'].squeeze(1).to(DEVICE)
        wrong_text_mask = content['Data_Mask'].to(DEVICE)

        text_generate = Model.generate(wrong_text_label, max_len = max_len)

        text_output_list = tokenizer.batch_decode(text_generate, skip_special_tokens = True)
        label_output_list = tokenizer.batch_decode(correct_text_label, skip_special_tokens = True)
        origin_content_list = tokenizer.batch_decode(wrong_text_label, skip_special_tokens = True)
        for i in range(0, correct_text_label.size()[0]):
            m2_output_file(text_output_list[i], label_output_list[i], origin_content_list[i], file_path + 'corrected.txt', file_path + 'label.txt', file_path+ 'origin.txt')
    ERRANT_evaluate(file_path, joint_training = joint_training)

    ### using ERRANT 



def test_ERRANT(output, label):
    annotator = errant.load('en')
    orig = annotator.parse(output)
    cor = annotator.parse(label)
    edits = annotator.annotate(orig, cor)
    for e in edits:
        print(e.o_start, e.o_end, e.o_str, e.c_start, e.c_end, e.c_str, e.type)
    return 0

def m2_output_file(output, label,origin,  output_filename, label_filename, origin_filename):

    with open(output_filename, 'a+') as f:
        f.write(output + '\n')
    with open(label_filename, 'a+') as f:
        f.write(label+ '\n')
    with open(origin_filename, 'a+') as f:
        f.write(origin + '\n')

def ERRANT_evaluate(file_path, joint_training):
    print('Using ERRANT to evaluate F0.5 score')
    print('Check Exist files')
    file_list = [ f for f in os.listdir(file_path) if  f.endswith('.m2')]
    for f in file_list:
        os.remove(os.path.join(file_path, f))

    print('Start evaluating')
    Org_Label = 'errant_parallel -orig ' + file_path + 'origin.txt ' +  '-cor ' + file_path +  'label.txt ' + '-out ' + file_path + 'origin_compare_label.m2 '     
    Text_Org = 'errant_parallel -orig ' + file_path + 'origin.txt ' +  '-cor ' + file_path +  'corrected.txt ' + '-out ' + file_path + 'corrected_compare_origin.m2 '
    os.system(Org_Label)
    os.system(Text_Org)
    Compare_commend = 'errant_compare -hyp ' + file_path + 'corrected_compare_origin.m2 ' + '-ref ' + file_path + 'origin_compare_label.m2'
    # os.system('errant_parallel -orig ./result/origin.txt -cor ./result/text.txt -out ./result/text_compare_origin.m2 ')
    steam = os.popen(Compare_commend)
    result = steam.read()
    with open( file_path + 'F_score_'+ joint_training + '.txt', 'w+') as f:
        f.write(result)

    


    
    


if __name__ == '__main__':
    # test_output = torch.ones((2,30))
    # test_label = torch.ones((2,30))
    # test_label[1,:25] = 0
    # test_output[1,:25] = 0
    # test_F = evaluate(test_output, test_label)
    # print(test_F)
    # test_output = "This are a correct sentences"
    # test_label = "This is a correct sentence"
    # test_origin = 'This is a correct sentences'
    # m2_output_file(test_output, test_label, test_origin, './result/text.txt', './result/label.txt', './result/origin.txt')
    # print('1')
    # stream = os.system('errant_parallel -orig ./result/origin.txt -cor ./result/label.txt -out ./result/origin_compare_label.m2 ')
    # print('2')
    # stream = os.system('errant_parallel -orig ./result/origin.txt -cor ./result/text.txt -out ./result/text_compare_origin.m2')
    # time.sleep(10)
    # print('3')
    # stream = os.popen('errant_compare -hyp ./result/text_compare_origin.m2 -ref ./result/origin_compare_label.m2')
    # print('end')
    # output = stream.read()
    # print(output)
    # errant_parallel -orig ./result/origin.txt -cor ./result/label.txt -out ./result/origin_compare_label.m2
    # errant_parallel -orig ./result/origin.txt -cor ./result/text.txt -out ./result/text_compare_origin.m2
    # !errant_compare -hyp ./result/text_compare_origin.m2 -ref ./result/origin_compare_label.m2
    ERRANT_evaluate('./result/')