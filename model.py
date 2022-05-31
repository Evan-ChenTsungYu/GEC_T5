
import torch.nn as nn
import torch
from transformers import T5Config,T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoConfig

class GEC_model(nn.Module):
    def __init__(self, pretrain_name, joint_training = 'none'):
        super().__init__()

        config = AutoConfig.from_pretrained(pretrain_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrain_name,
        config = config
        )
        self.joint_training = joint_training
        if self.joint_training == 'tag':
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64,1),
                nn.Sigmoid()
            )
        elif self.joint_training == 'type':
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64,56),
                nn.Softmax(dim = 2)
            )
        

    def forward(self, input_data,input_mask, text_label, seperate_training = 'None'):
        if seperate_training == 'encoder':
            self.model.decoder.requires_grad = False
            self.model.lm_head.requires_grad = False
        elif seperate_training == 'decoder':
            self.model.decoder.requires_grad = True
            self.model.lm_head.requires_grad = True


        text_output = self.model(input_data, attention_mask =input_mask, labels = text_label)
        if self.joint_training == 'tag':
            class_output = self.classifier(text_output.encoder_last_hidden_state)
            return torch.transpose(text_output.logits, 1,2), class_output.squeeze(2)
        elif self.joint_training == 'type':
            class_output = self.classifier(text_output.encoder_last_hidden_state)
            return torch.transpose(text_output.logits, 1,2), torch.transpose(class_output,1,2)
        else: 
            return torch.transpose(text_output.logits, 1,2), None
        
    def generate(self, input_data, max_len):
        return self.model.generate(input_data, max_length = max_len, num_beams = 8)
        
        
        
if __name__ == '__main__':
    DEVICE = 'cuda:2'
    test_model = GEC_model('t5-base').to(DEVICE)
    import torch
    test_data = torch.randint(0,200, (3,200)).to(DEVICE)
    test_label = torch.randint(0,200, (3,200)).to(DEVICE)
    test_mask = torch.ones((3,200)).to(DEVICE)
    test_T5output,test_classoutput = test_model(test_data, test_mask, test_label)
    print(test_T5output.shape,test_classoutput.shape)
    test_generate = test_model.generate(test_data)
    print(test_generate)