import torch.nn as nn
from tqdm import tqdm 
from loss_fn import joint_loss
import torch
def train(model, TT_DL,Val_DL, epochs, result_save_path, tokenizer, show_epoch_result = True, DEVICE = 'cpu', joint_training = 'none'):
    print(f'Train on {len(TT_DL)} sentence, with Validation set be {len(Val_DL)} sentence')
    start_epoch = 0 
    mini_val_loss = 1e10
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)
    Softmax = nn.Softmax(dim = 2)
    for epoch in tqdm(range(start_epoch, epochs)):
        running_loss = 0.0
        running_text_loss = 0.0
        running_class_loss = 0.0

        for index, content in enumerate(TT_DL):
            tag_label = content['error_tag'].squeeze(1).to(DEVICE)
            type_label = content['error_type'].squeeze(1).to(DEVICE)
            text_label = content['text_label'].squeeze(1).to(DEVICE)
            text_mask = content['text_mask'].squeeze(1).to(DEVICE)
            input_data = content['Data'].squeeze(1).to(DEVICE)
            input_mask = content['Data_Mask'].to(DEVICE)
            optimizer.zero_grad()

            text_output, class_output = model(input_data, input_mask, text_label, seperate_training = 'decoder')
            if joint_training == 'tag':
                class_label = tag_label
                class_label = class_label.float()
            elif joint_training == 'type':
                class_label = type_label.long()
            else:
                class_label = torch.ones(tag_label.size()).to(DEVICE)
            
            text_loss, class_loss = joint_loss(class_output = class_output, class_label = class_label, text_output = text_output, text_label = text_label, joint_training = joint_training)
            total_loss = text_loss.to(DEVICE) + class_loss.to(DEVICE)
            
            text_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()/len(TT_DL)
            running_text_loss += text_loss.item()/len(TT_DL)
            running_class_loss += class_loss.item()/len(TT_DL)

            ## Val Dataset  ###############################################################################
        scheduler.step()
        val_loss = 0
        print(f'Test Validation Loss, set val loss = {val_loss}')
        with torch.no_grad():
            running_val_loss = 0.0
            for val_index, val_content in enumerate(Val_DL):
                # val_class_label = val_content['classify_label'].squeeze(1).to(DEVICE)
                val_tag_label = val_content['error_tag'].squeeze(1).to(DEVICE)
                val_type_label = val_content['error_type'].squeeze(1).to(DEVICE)

                val_text_label = val_content['text_label'].squeeze(1).to(DEVICE)
                val_input_data = val_content['Data'].squeeze(1).to(DEVICE)
                val_input_mask = val_content['Data_Mask'].to(DEVICE)
                val_text_output, val_class_output = model(val_input_data, val_input_mask, val_text_label)
                if joint_training == 'tag':
                    val_class_label = val_tag_label
                    val_class_label = val_class_label.float()
                elif joint_training == 'type':
                    val_class_label = val_type_label.long()
                else:
                    val_class_label = torch.zeros(tag_label.size()).to(DEVICE)

                val_text_loss, val_class_loss = joint_loss(class_output = val_class_output, class_label = val_class_label, text_output = val_text_output, text_label = val_text_label, joint_training = 'None')
                running_val_loss = val_text_loss.to(DEVICE) + val_class_loss.to(DEVICE)

            print(f'In each epoch check the output of decoder :')
            print(f'output : {tokenizer.decode(torch.argmax(val_text_output, dim = 1)[0,:], skip_special_tokens = True)}')
            print(f'labels : {tokenizer.decode(val_text_label[0,:], skip_special_tokens = True)}')
            if running_val_loss <= mini_val_loss:
                print(f'renew val_loss: {running_val_loss/len(Val_DL)}')
                mini_val_loss = running_val_loss
                torch.save(model.state_dict(), result_save_path)
        
        if show_epoch_result :
            print(f'[{epoch}/ {epochs-1}], loss:{running_loss}, text_loss :{running_text_loss}, class_loss:{running_class_loss} / val_loss:{running_val_loss/len(Val_DL)}') 
        