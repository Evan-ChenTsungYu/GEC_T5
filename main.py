import argparse
from genericpath import exists
import torch
import json
from model import GEC_model
from  make_dataset import load_GEC_dataset, GEC_DS
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from evaluate import evaluate
from train import train
from parameter import param #define each parameters. 
from store_result import result_json
from pretrain_Enc import pretrain_Enc
import os
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-tti',  type = str, help='input train/test', dest = "TT") #現在要train/test/inference
    parser.add_argument('-e', type = int, help = 'epoch to train', dest="epoch")
    parser.add_argument('-pe', type = int, help = 'epoch to pretrain', dest= "pretrain_epoch")
    parser.add_argument('-g', type = int, help = 'Which GPU to used', dest = 'GPU')
    args, remaining_args = parser.parse_known_args() #args 是存已知的args, remain 是存使用者不小心輸錯的
    
    DEVICE = 'cuda:'+str(args.GPU)

    parameter = param()

    print(parameter.result_save_path)

    tokenizer = AutoTokenizer.from_pretrained(parameter.model_name)

    dataset_list = parameter.dataset_list
    TT_DL, Val_DL = load_GEC_dataset(dataset_list, args.TT , parameter.batch_size, parameter.split_size, parameter.val_split_size) #load_dataset with train/test/inference
    Model = GEC_model(parameter.model_name, parameter.joint_training).to(DEVICE)
    Model.train()
    if args.TT == 'train':
        print(f' joint training type :{parameter.joint_training}')
        # if parameter.joint_training != 'none' and os.path.exists(parameter.pretrain_save_path) == False:
        #     print('start pre_training on encoder')
        #     pretrain_Enc(Model, TT_DL, Val_DL, epochs = args.pretrain_epoch, result_save_path = parameter.pretrain_save_path, tokenizer = tokenizer, show_epoch_result = True, DEVICE = DEVICE, joint_training = parameter.joint_training)
        # elif os.path.exists(parameter.pretrain_save_path):
        #     print('Load Pretrain Weight')
        #     Model.load_state_dict(torch.load(parameter.pretrain_save_path, map_location = DEVICE))
        # print('start training')
        # train(Model, TT_DL, Val_DL, args.epoch, parameter.result_save_path,tokenizer, show_epoch_result = True, DEVICE = DEVICE, joint_training = parameter.joint_training)
        print(f'find lang8 pretrained : {parameter.lang8_pretrained}')
        Model.load_state_dict(torch.load(parameter.lang8_pretrained, map_location = DEVICE))
        if parameter.joint_training != 'none':
            print(f'start pretrain and store in {parameter.pretrain_save_path}')
            pretrain_Enc(Model, TT_DL, Val_DL, epochs = pretrain_epoch, result_save_path = parameter.pretrain_save_path, tokenizer = tokenizer, show_epoch_result = True, DEVICE = DEVICE, joint_training = parameter.joint_training)
        print(f'start training and store in {parameter.result_save_path}')
        train(Model, TT_DL, Val_DL, args.epoch, parameter.result_save_path,tokenizer, show_epoch_result = True, DEVICE = DEVICE, joint_training = parameter.joint_training)
        
    else:
        print(f'start testing on {dataset_list[args.TT]} with {parameter.batch_size*len(TT_DL)} sentence')
        param_dict = parameter.parameter_dict()
        Model.load_state_dict(torch.load(parameter.result_save_path, map_location = DEVICE))
        # Model.load_state_dict(torch.load('./model_save/model_lang8_pretrain.pt', map_location = DEVICE))
        Model.eval()
        evaluate(Model, TT_DL, parameter.max_len, DEVICE, tokenizer, parameter.joint_training)
        
        # result_json(result, param_dict, './result'+'.json')
    return 0


if __name__ == "__main__" :
    main()