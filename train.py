# simplified training script updates: it already writes meta json
import argparse
import os
from ml.classifier import Classifier
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='trashnet-master/data/dataset-resized')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--out', default='models_weights/trash_classifier.pth')
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    print('Dataset not found at', args.data_dir)
    exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Training on', device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(args.data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

clf = Classifier(model_path=None, device=device)
model = clf.model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    model.train()
    total=0
    correct=0
    running_loss=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
    acc = correct/total if total>0 else 0
    print(f'Epoch {epoch+1}/{args.epochs} loss={running_loss:.4f} acc={acc:.4f}')

# save
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save(model.state_dict(), args.out)
meta = {'accuracy': acc}
with open(os.path.splitext(args.out)[0]+'.json','w',encoding='utf-8') as f:
    json.dump(meta,f)
print('Saved model to', args.out)
