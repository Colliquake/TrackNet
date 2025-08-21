import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TennisDataset
from model import TrackNet
from train import train
from validate import validate

if __name__ == "__main__":

    device = 'cuda'

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    batch_size = 2

    base_path = './datasets/tennis'
    full_dataset = TennisDataset(base_path, frames=3)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = TrackNet(frames=3, out_channels=256).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    epochs = 100

    e = 0

    for epoch in range(e, epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        print(f"epoch {epoch}, training loss: {train_loss}")
        val_loss, avg_dist, tp, fp, tn, fn, total_visibility = validate(model, val_dataloader, criterion, device)
        # val_loss = float("{:.2f}".format(val_loss))
        # avg_dist = float("{:.2f}".format(avg_dist))
        print(f"epoch: {epoch}, val loss: {val_loss}, avg_dist: {avg_dist}")
        print(f"tp: {tp}")
        print(f"fp: {fp}")
        print(f"tn: {tn}")
        print(f"fn: {fn}")
        print(f"tv: {total_visibility}")
        tp_total = sum(tp)
        fp_total = sum(fp)
        tn_total = sum(tn)
        fn_total = sum(fn)

        precision = tp_total / (tp_total + fp_total + 1)
        recall = tp_total / (sum(tp[-3:]) + sum(fp[-3:]) + sum(tn[-3:]) + sum(fn[-3:]) + 1)
        f1 = (2 * precision * recall) / (precision + recall + 1)
            
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")
        if f1 > best_f1:
            print("better model detected. saving model.")
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'f1': f1
            }, 'model.pth')

        if epoch % 2 == 0:
            print("saving...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'f1': f1
            }, 'current_model.pth')