
        '''def forward_pass():
            scores = model(imgs)
            loss = loss_fn(scores, labels)
            return loss'''

        '''if device == 'cuda':
            with torch.cuda.amp.autocast():
                loss = forward_pass()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()'''
