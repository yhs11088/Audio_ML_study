# source : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

def train_VAE(model, optimizer, loss_fn, train_loader, device,
              batch_size = 16, input_dim = 784,
              epochs = 30):
    
    model.train()

    for epoch in range(epochs):
        total_loss = 0.
        n_data = 0

        for batch_idx, (x, _) in enumerate(train_loader):
            
            B, C, H, W = x.shape # shape of x : (batch, channel, H, W)

            x = x.view(B*C, H*W)
            #x = x.view(batch_size, input_dim)
            x = x.to(device)

            x_hat, mean, log_var = model(x)
            loss = loss_fn(x, x_hat, mean, log_var)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_data += B

        #print(f"  Epoch {epoch+1} : Average Loss = {total_loss / (batch_idx * batch_size)}")
        print(f"  Epoch {epoch+1} : Average Loss = {total_loss / n_data:.2f}")