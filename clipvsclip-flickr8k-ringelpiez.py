import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import clip
from colorama import Fore, Style, init
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore') 


init(autoreset=True)#colorama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to CLIP's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Flickr8k Dataset Class
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_captions = self.load_captions(captions_file)

    def load_captions(self, captions_file):
        image_captions = []
        with open(captions_file, 'r') as file:
            for line in file:
                image_file, caption = line.strip().split(',', 1)
                image_path = os.path.join(self.img_dir, image_file)
                image_captions.append((image_path, caption))
        return image_captions

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        img_path, caption = self.image_captions[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption


# Load Flickr8k dataset
data_dir = "path/to/Flickr8k/Images"
captions_file = "labels/flickr8k_val_karpathy.txt"
flickr_dataset = Flickr8kDataset(img_dir=data_dir, captions_file=captions_file, transform=transform)

# Generate text prompts from captions
texts = [caption for _, caption in flickr_dataset.image_captions]

# Create data loader
data_loader = DataLoader(flickr_dataset, batch_size=32, shuffle=True)

# Load models (pre-trained and fine-tuned)
pretrained_model, preprocess = clip.load("ViT-L/14", device=device)
fine_tuned_model, _ = clip.load("ViT-L/14", device=device)
fine_tuned_model.load_state_dict(torch.load("finetune/ViT-L-14-BEST-smooth-GmP-ft-state_dict.pt", map_location="cuda"))

# Manually swap the text encoders and image encoders
class CustomCLIPModel(torch.nn.Module):
    def __init__(self, visual_encoder, text_encoder):
        super().__init__()
        self.visual = visual_encoder
        self.transformer = text_encoder
        self.token_embedding = pretrained_model.token_embedding  # Shared embedding layer
        self.positional_embedding = pretrained_model.positional_embedding
        self.ln_final = pretrained_model.ln_final
        self.text_projection = pretrained_model.text_projection
        self.logit_scale = pretrained_model.logit_scale
        self.dtype = pretrained_model.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Take features from the [EOS] token (which is the highest number of context tokens)
        eos_token_idx = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_token_idx] @ self.text_projection

        return x

# Scenario 1: Pre-trained Image Encoder + Fine-tuned Text Encoder
pretrained_image_finetuned_text_model = CustomCLIPModel(pretrained_model.visual, fine_tuned_model.transformer).to(device)

# Scenario 2: Fine-tuned Image Encoder + Pre-trained Text Encoder
finetuned_image_pretrained_text_model = CustomCLIPModel(fine_tuned_model.visual, pretrained_model.transformer).to(device)

# Tokenize the text inputs using CLIP
text_tokens = clip.tokenize(texts).to(device)

# Function to perform cross-modal retrieval (image to text or text to image)
def cross_modal_retrieval(images, text_tokens, model):
    with torch.no_grad():
        # Encode images and texts
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        # Normalize features to unit vectors
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity matrix between images and texts
        similarity_matrix = image_features @ text_features.T

        # Sort and retrieve results
        retrieval_results = similarity_matrix.argsort(dim=-1, descending=True)
        return retrieval_results

# Function to evaluate retrieval performance
def evaluate_model_retrieval(data_loader, text_tokens, model_name, model):
    class_correct = 0
    total_images = 0
    for images, captions in data_loader:
        images = images.to(device)

        # Perform cross-modal retrieval for the current model
        retrieval_results = cross_modal_retrieval(images, text_tokens, model)

        # Check how many top-1 results match the ground-truth captions
        for i, caption in enumerate(captions):
            caption_index = texts.index(caption)  # Get the index of the current caption in the text_tokens
            if retrieval_results[i, 0].item() == caption_index:
                class_correct += 1
            total_images += 1

    accuracy = class_correct / total_images * 100
    print(f"{model_name} accuracy: {accuracy:.2f}% ({class_correct}/{total_images})")
    return accuracy

# Evaluate the original models and the swapped models
print(f"{Fore.LIGHTYELLOW_EX}Evaluating Pre-trained CLIP model:")
evaluate_model_retrieval(data_loader, text_tokens, "Pre-trained CLIP", pretrained_model)

print(f"{Fore.LIGHTGREEN_EX}Evaluating Fine-tuned CLIP model:")
evaluate_model_retrieval(data_loader, text_tokens, "Fine-tuned CLIP", fine_tuned_model)

print(f"{Fore.CYAN}Evaluating Pre-trained Image Encoder + Fine-tuned Text Encoder:")
evaluate_model_retrieval(data_loader, text_tokens, "Pre-trained Image + Fine-tuned Text", pretrained_image_finetuned_text_model)

print(f"{Fore.MAGENTA}Evaluating Fine-tuned Image Encoder + Pre-trained Text Encoder:")
evaluate_model_retrieval(data_loader, text_tokens, "Fine-tuned Image + Pre-trained Text", finetuned_image_pretrained_text_model)
