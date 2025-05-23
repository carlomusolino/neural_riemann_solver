import torch 
import numpy as np 
import os 
import torch.optim
from .PCGrad_utils import PCGrad

def train_model_PCGrad(model, optimizer, scheduler, train_loader, test_loader, epochs, checkpoint_dir, n_components, loss_fn, clip_grads=False, max_grad_norm=1.0, *loss_args):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    optimizer = PCGrad(optimizer)

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = [0 for i in range(n_components)]

        for batch in train_loader:
            batch_features, batch_labels = batch
            optimizer.zero_grad()

            losses = loss_fn(model, batch_features, batch_labels, *loss_args)
            optimizer.pc_backward(losses)
            # Optionally clip gradients to prevent instability 
            if clip_grads:
                parameters = [p for p in model.parameters() if p.grad is not None]
                if parameters:
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)

            # Optimizer step
            optimizer.step()

            # Accumulate total training loss
            for i,loss in enumerate(losses):
                epoch_train_losses[i] += loss.item() * batch_features.size(0)

        # Normalize by dataset size
        train_losses.append([ epoch_train_loss / len(train_loader.dataset) for epoch_train_loss in epoch_train_losses ])
        epoch_train_loss = sum(epoch_train_losses) / len(train_loader.dataset)
        # ---------------- VALIDATION ----------------
        model.eval()
        epoch_test_losses = [0 for i in range(n_components)]
        with torch.no_grad():
            for batch in test_loader:
                batch_features, batch_labels = batch
                losses = loss_fn(model, batch_features, batch_labels, *loss_args)
                # Accumulate total training loss
                for i,loss in enumerate(losses):
                    epoch_test_losses[i] += loss.item() * batch_features.size(0)
        test_losses.append([epoch_test_loss/len(test_loader.dataset) for epoch_test_loss in epoch_test_losses])
        epoch_test_loss = sum(epoch_test_losses) / len(test_loader.dataset)
        # Scheduler step based on validation loss
        if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau): 
            scheduler.step(epoch_test_loss)
        else:
            scheduler.step()
        # Checkpoint if best
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Epoch {epoch+1}] ✅ Best model saved with test loss: {best_test_loss:.6f}")

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6e}")

    return train_losses, test_losses