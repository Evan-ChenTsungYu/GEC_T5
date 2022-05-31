from dataclasses import dataclass
import os

@dataclass
class param():
    model_name = 't5-base'
    batch_size = 32
    max_len = 100
    split_size = 1
    val_split_size = 1
    joint_training = 'none'
    dataset_list = {'train': ['wi', 'fce'], 'test':['conll'], 'valid':['wi', 'conll']}
    lang8_pretrained = './model_save/model_'+ joint_training + '_ds_' + 'lang8' + '.pt'
    
    if os.path.exists(lang8_pretrained) == False:
        result_save_path = './model_save/model_'+ joint_training + '_ds_' + '_'.join(dataset_list['train']) + '.pt'
        pretrain_save_path = './model_save/model_'+ joint_training + '_ds_' + '_'.join(dataset_list['train']) + '_pretrain.pt'
    else:
        result_save_path = './model_save/model_'+ joint_training + '_ds_lang8_' + '_'.join(dataset_list['train']) + '.pt'
        pretrain_save_path = './model_save/model_'+ joint_training + '_ds_lang8_' + '_'.join(dataset_list['train']) + '_pretrain.pt'
    
    # result_save_path = './model_save/model_'+ joint_training + '_ds_' + '_'.join(dataset_list['train']) + '.pt'
    # pretrain_save_path = './model_save/model_'+ joint_training + '_ds_' + '_'.join(dataset_list['train']) + '_pretrain.pt'
    def parameter_dict(self):
        parameter = {'parameter': {'model_name':self.model_name, 'batch_size':self.batch_size, 'max_len':self.max_len, 'split_size':self.split_size}}
        return parameter

    
if __name__ == '__main__':
    test_parm = param()
    print(test_parm.parameter_dict())