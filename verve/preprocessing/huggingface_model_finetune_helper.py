import torch
from torch.utils.data import Dataset


def train(epoch, tokenizer, model, device, loader, val_loader, optimizer, testing=True):
    model.train()
    running_loss = 0.0
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids,
                        attention_mask=mask,
                        decoder_input_ids=y_ids,
                        lm_labels=lm_labels)
        loss = outputs[0]

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss_val = 0.0
    if testing:
        with torch.no_grad():
            for _, data in enumerate(val_loader, 0):
                y = data['target_ids'].to(device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
                ids = data['source_ids'].to(device, dtype=torch.long)
                mask = data['source_mask'].to(device, dtype=torch.long)

                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                    decoder_input_ids=y_ids,
                    lm_labels=lm_labels)
                loss = outputs[0]

                running_loss_val += loss.item()

        return running_loss / len(loader), running_loss_val / len(val_loader)
    else:
        return running_loss / len(loader), 0


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext],
                                                  max_length=self.source_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text],
                                                  max_length=self.summ_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def inference(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [
                tokenizer.decode(
                    g,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [
                tokenizer.decode(
                    t,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True)for t in y]

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals
