import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights='weights/best.pt'
# Load model
model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
#model.to(device).eval()
torch.save(torch.load(weights, map_location=device)['model'],'best_model.pt')