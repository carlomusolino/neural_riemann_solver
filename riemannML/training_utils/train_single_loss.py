import torch 
import numpy as np 
import os 

def train_model(model, optimizer, scheduler, train_loader, test_loader, epochs, checkpoint_dir, loss_fn, *loss_args):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            batch_features, batch_labels = batch
            optimizer.zero_grad()
            
            loss = loss_fn(model, batch_features, batch_labels, *loss_args)
            loss.backward()
            # Clip gradients to prevent instability 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_features.size(0)  # Sum loss over batch
        
        if np.isnan(epoch_train_loss): break 
        
        epoch_train_loss /= len(train_loader.dataset)  # Average loss
        train_losses.append(epoch_train_loss)

        # ------------------ VALIDATION ------------------
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch_features, batch_labels = batch
                loss = loss_fn(model, batch_features, batch_labels, *loss_args)
                epoch_test_loss += loss.item() * batch_features.size(0)
        
        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        # ------------------ Scheduler Step ------------------
        scheduler.step(epoch_test_loss)  # Adjust LR based on validation loss

        # ------------------ Checkpointing ------------------
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            #print(f"[Epoch {epoch+1}] ✅ Best model saved with test loss: {best_test_loss:.6f}")

        # ------------------ Logging ------------------
        if (epoch+1)%10 == 0 or epoch==0: 
            print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}")
    return train_losses, test_losses 