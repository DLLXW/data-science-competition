import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch import nn
from torch.cuda.amp import autocast, GradScaler
def train_fn(model,data_loader, optimizer, scheduler, epoch, device,criterion,scaler):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Training epoch: " + str(epoch+1))

    for t,data in enumerate(tk):
        optimizer.zero_grad()
        for k,v in data.items():
            data[k] = v.to(device)
        # output=model(**data)
        # loss=criterion(output,data['label'])
        # loss.backward()
        # optimizer.step()
        with autocast():
            output=model(**data)
            loss=criterion(output,data['label'])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        fin_loss += loss.item() 

        tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1)), 'LR' : optimizer.param_groups[0]['lr']})

    scheduler.step()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, epoch, device,criterion,scaler):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Validation epoch: " + str(epoch+1))

    with torch.no_grad():
        for t,data in enumerate(tk):
            for k,v in data.items():
                data[k] = v.to(device)

            with autocast():
                output=model(**data)
                loss=criterion(output,data['label'])
            fin_loss += loss.item() 

            tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1))})
        return fin_loss / len(data_loader)
