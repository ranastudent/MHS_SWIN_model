from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def preprocess(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)