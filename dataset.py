from torch.utils.data import Dataset
import json

class BugDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.samples = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code = self.samples[idx]['func']
        label = self.samples[idx]['target']
        inputs = self.tokenizer(code, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = label
        return inputs
