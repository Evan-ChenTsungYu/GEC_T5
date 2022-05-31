import torch
import torch.nn as nn
def joint_loss(class_output,class_label, text_output, text_label, joint_training):
    
    CELcriterion = nn.CrossEntropyLoss(ignore_index = 0)
    # TypeCriterion = nn.CrossEntropyLoss(ignore_index= 0 )
    BELcriterion = nn.BCELoss()

    if joint_training == 'tag':
        text_loss = CELcriterion(text_output, text_label)
        class_loss = BELcriterion(class_output, class_label)
    elif joint_training == 'type':
        text_loss = CELcriterion(text_output, text_label)
        class_loss = CELcriterion(class_output, class_label)
    else:
        text_loss = CELcriterion(text_output, text_label)
        class_loss = torch.FloatTensor([0.0])

    return text_loss, class_loss
    