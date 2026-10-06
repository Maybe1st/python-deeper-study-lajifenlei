import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import os
from config import CLASS_NAMES

class Classifier:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.num_classes = len(CLASS_NAMES)
        self._build_model()
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print('Loaded model weights:', model_path)
            except Exception as e:
                print('Failed to load model weights:', e)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def _build_model(self):
        import torchvision.models as models
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, self.num_classes)
        self.model = m

    def predict(self, pil_image, topk=3):
        img = pil_image.convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        import numpy as np
        topidx = probs.argsort()[::-1][:topk]
        results = []
        for i in topidx:
            results.append({
                'class_name': CLASS_NAMES[i],
                'class_cn': None,
                'confidence': float(probs[i])
            })
        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
