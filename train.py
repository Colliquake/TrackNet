from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    
    for iter_id, (imgs, gt_output, label) in enumerate(train_loader):
        imgs = imgs.to(device)
        gt_output = gt_output.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        gt_output = gt_output.long()
        loss = criterion(outputs, gt_output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss / (iter_id + 1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update()

    batch_bar.close()
    return running_loss / len(train_loader)
