import timm
from config import CFG
model_name=CFG.model_name
backbone = timm.create_model(model_name, pretrained=False)
print(backbone.classifier.in_features)