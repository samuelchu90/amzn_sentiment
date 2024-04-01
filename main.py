from datasets import load_dataset
from torch.utils.data import DataLoader

train = load_dataset("csv", data_files="intent_train.csv")
val = load_dataset("csv", data_files="intent_val.csv")
test = load_dataset("csv", data_files="intent_test.csv")
print(train, val, test)


#print(val['train'][0])
dataloader = DataLoader(train['train'], batch_size=4)
for batch in dataloader:
    print(batch)