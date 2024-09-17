import os
import torch
import torch.nn.functional as F
import numpy as np
import clip
from clip.model import QuickGELU
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
import warnings

'''
Warning: This code is awful, redundant, confusing, and cringy.
Programmers: Viewer discretion is advised. Trigger warning and all that.

But it works.
It's good enough for an arbitrary experiment.
'''


warnings.filterwarnings('ignore') # Eliminate spam about future pickle warning with latest torch
init(autoreset=True) # colorama

results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# Load CLIP models (pre-trained and fine-tuned)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model, preprocess = clip.load("ViT-L/14", device=device)
fine_tuned_model, _ = clip.load("ViT-L/14", device=device)
fine_tuned_model.load_state_dict(torch.load("finetune/ViT-L-14-BEST-smooth-GmP-ft-state_dict.pt", map_location="cuda"))

pretrained_model = pretrained_model.float()
fine_tuned_model = fine_tuned_model.float()

class ClipNeuronCaptureHook:
    def __init__(self, module: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.detach()  # Capture post-GELU activations

    def get_activations(self):
        if self.activations is not None:
            return self.activations.cpu().numpy()
        return None

def register_hooks_for_model(model, num_layers, transformer_type="vision"):
    hooks = []
    layer_idx = 0
    if transformer_type == "vision":
        for name, module in model.visual.named_modules():  # Use the vision transformer
            if isinstance(module, QuickGELU):
                hook = ClipNeuronCaptureHook(module, layer_idx)
                hooks.append(hook)
                layer_idx += 1
                if layer_idx >= num_layers:
                    break
    elif transformer_type == "text":
        for name, module in model.transformer.named_modules():  # Use the text transformer
            if isinstance(module, QuickGELU):
                hook = ClipNeuronCaptureHook(module, layer_idx)
                hooks.append(hook)
                layer_idx += 1
                if layer_idx >= num_layers:
                    break
    return hooks

# PCA on the activations
def perform_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

# Compute KL Divergence between two distributions
def kl_divergence(p, q):
    p = p + 1e-10  # To avoid log(0)
    q = q + 1e-10
    return torch.sum(p * torch.log(p / q))

# KL Divergence between activations of pre-trained and fine-tuned
def compute_kl_for_features(pretrained, fine_tuned):
    kl_divs = []
    for i in range(pretrained.shape[1]):  # Iterate over 4096 features
        p = pretrained[:, i]  # Pre-trained activations for feature i
        q = fine_tuned[:, i]  # Fine-tuned activations for feature i
        kl_div = kl_divergence(F.softmax(p, dim=0), F.softmax(q, dim=0))
        kl_divs.append(kl_div.item())
    return kl_divs


'''
Set the layer of interest for analysis below!
num_layers_vision-1 or num_layers_text-1 = last layer
'''

def detect_num_layers_and_default_layers(model):
    num_layers_vision = 0
    num_layers_text = 0
    layer_idx_vision = None
    layer_idx_text = None

    # vision transformer
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers_vision = len(model.visual.transformer.resblocks)
        layer_idx_vision = num_layers_vision-2  # Default to last_block-2 for vision (penultimate layer)

    # text transformer
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
        num_layers_text = len(model.transformer.resblocks)
        layer_idx_text = num_layers_text-2  # Default to last_block-2 for text (penultimate layer)

    return num_layers_vision, layer_idx_vision, num_layers_text, layer_idx_text


# Function to run the model and collect activations
def run_model_and_collect_activations(model, input_data, hooks, mode="vision"):
    if mode == "vision":
        _ = model.encode_image(input_data)  # Vision transformer forward pass
    elif mode == "text":
        _ = model.encode_text(input_data)  # Text transformer forward pass
    # Collect activations for all hooked layers
    activations = [hook.get_activations() for hook in hooks]
    return activations

def plot_pca_separate(pca_vision_pretrained, pca_vision_finetuned, pca_text_pretrained, pca_text_finetuned,
                      var_vision_pretrained, var_vision_finetuned, var_text_pretrained, var_text_finetuned, imagename):
    # Plot for Vision Transformer
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_vision_pretrained[:, 0], pca_vision_pretrained[:, 1], color='blue', label='Pre-trained Vision activations', alpha=0.5)
    plt.scatter(pca_vision_finetuned[:, 0], pca_vision_finetuned[:, 1], color='green', label='Fine-tuned Vision activations', alpha=0.5)
    plt.title(f"{imagename} PCA Comparison - Vision Transformer (Variance: Pre-trained={var_vision_pretrained.sum():.2f}, Fine-tuned={var_vision_finetuned.sum():.2f})")
    plt.xlabel(f"PC 1 ({var_vision_pretrained[0]:.2%})")
    plt.ylabel(f"PC 2 ({var_vision_pretrained[1]:.2%})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{imagename}-PCA_Vision_CLIP.png")
    plt.close()

    # Plot for Text Transformer
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_text_pretrained[:, 0], pca_text_pretrained[:, 1], color='red', label='Pre-trained Text activations', alpha=0.5)
    plt.scatter(pca_text_finetuned[:, 0], pca_text_finetuned[:, 1], color='orange', label='Fine-tuned Text activations', alpha=0.5)
    plt.title(f"{imagename} PCA Comparison - Text Transformer (Variance: Pre-trained={var_text_pretrained.sum():.2f}, Fine-tuned={var_text_finetuned.sum():.2f})")
    plt.xlabel(f"PC 1 ({var_text_pretrained[0]:.2%})")
    plt.ylabel(f"PC 2 ({var_text_pretrained[1]:.2%})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{imagename}-PCA_Text_CLIP.png")
    plt.close()

def color_divs(kl_divs):
    if kl_divs <= 1.00:
        return f"{Fore.LIGHTGREEN_EX}{kl_divs:.4f}{Style.RESET_ALL}"
    elif kl_divs <= 2.00:
        return f"{Fore.LIGHTYELLOW_EX}{kl_divs:.4f}{Style.RESET_ALL}"
    else:
        return f"{Fore.LIGHTRED_EX}{kl_divs:.4f}{Style.RESET_ALL}"


# Automatically detect the number of layers and the default layer indices to ensure compatibility with all models
num_layers_vision, default_layer_idx_vision, num_layers_text, default_layer_idx_text = detect_num_layers_and_default_layers(pretrained_model)

print('\n')
print(f"Using layer {default_layer_idx_vision} for vision analysis.")
print(f"Using layer {default_layer_idx_text} for text analysis.")

# Register hooks automatically based on the detected number of layers
pretrained_hooks_vision = register_hooks_for_model(pretrained_model, num_layers=num_layers_vision, transformer_type="vision")
pretrained_hooks_text = register_hooks_for_model(pretrained_model, num_layers=num_layers_text, transformer_type="text")

fine_tuned_hooks_vision = register_hooks_for_model(fine_tuned_model, num_layers=num_layers_vision, transformer_type="vision")
fine_tuned_hooks_text = register_hooks_for_model(fine_tuned_model, num_layers=num_layers_text, transformer_type="text")

# Load and preprocess the image
image_path = "images/bwcat_dog.png"
imagename = os.path.splitext(os.path.basename(image_path))[0]
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Text for the text encoder
text_input = clip.tokenize(["a photo of a cat"]).to(device)

# Run the models and capture activations for both text and vision transformers
pretrained_act_vision = run_model_and_collect_activations(pretrained_model, image, pretrained_hooks_vision, mode="vision")
fine_tuned_act_vision = run_model_and_collect_activations(fine_tuned_model, image, fine_tuned_hooks_vision, mode="vision")

pretrained_act_text = run_model_and_collect_activations(pretrained_model, text_input, pretrained_hooks_text, mode="text")
fine_tuned_act_text = run_model_and_collect_activations(fine_tuned_model, text_input, fine_tuned_hooks_text, mode="text")

layer_idx_vision = default_layer_idx_vision
layer_idx_text = default_layer_idx_text

# Extract features
pretrained_features_vision = torch.tensor(pretrained_act_vision[layer_idx_vision]).squeeze(1)
fine_tuned_features_vision = torch.tensor(fine_tuned_act_vision[layer_idx_vision]).squeeze(1)

pretrained_features_text = torch.tensor(pretrained_act_text[layer_idx_text]).squeeze(1)
fine_tuned_features_text = torch.tensor(fine_tuned_act_text[layer_idx_text]).squeeze(1)

# Convert to numpy arrays for PCA
pretrained_np_vision = pretrained_features_vision.cpu().numpy()
fine_tuned_np_vision = fine_tuned_features_vision.cpu().numpy()

pretrained_np_text = pretrained_features_text.cpu().numpy()
fine_tuned_np_text = fine_tuned_features_text.cpu().numpy()

pca_pretrained_vision, var_pretrained_vision = perform_pca(pretrained_np_vision)
pca_finetuned_vision, var_finetuned_vision = perform_pca(fine_tuned_np_vision)

pca_pretrained_text, var_pretrained_text = perform_pca(pretrained_np_text)
pca_finetuned_text, var_finetuned_text = perform_pca(fine_tuned_np_text)

# Plot PCA results for vision and text transformers separately
plot_pca_separate(pca_pretrained_vision, pca_finetuned_vision, pca_pretrained_text, pca_finetuned_text,
                  var_pretrained_vision, var_finetuned_vision, var_pretrained_text, var_finetuned_text, imagename)

# Continue with KL divergence analysis for vision and text transformers separately
kl_div_vision = compute_kl_for_features(pretrained_features_vision, fine_tuned_features_vision)
kl_div_text = compute_kl_for_features(pretrained_features_text, fine_tuned_features_text)


# Plot KL Divergence for each transformer
plt.figure(figsize=(10, 6))
plt.plot(kl_div_vision, label="KL Divergence - Vision")
plt.title(f"{imagename} KL Div -- Pre-trained vs Fine-tuned (Vision)")
plt.xlabel("Feature Index")
plt.ylabel("KL Divergence")
plt.legend()
plt.grid(True)
plt.savefig(f"results/{imagename}-KL_Vision_CLIP.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(kl_div_text, label="KL Divergence - Text")
plt.title(f"{imagename} KL Div -- Pre-trained vs Fine-tuned (Text)")
plt.xlabel("Feature Index")
plt.ylabel("KL Divergence")
plt.legend()
plt.grid(True)
plt.savefig(f"results/{imagename}-KL_Text_CLIP.png")
plt.close()

# KL Divergence between activations of pre-trained and fine-tuned - text threshold 0.1
def compute_kl_for_features(pretrained, fine_tuned, threshold=0.1):
    kl_divs = []
    high_kl_features = []
    for i in range(pretrained.shape[1]):
        p = pretrained[:, i]
        q = fine_tuned[:, i]
        kl_div = kl_divergence(F.softmax(p, dim=0), F.softmax(q, dim=0))
        if kl_div.item() >= threshold:  # Check if KL divergence exceeds the threshold
            high_kl_features.append(i)
        kl_divs.append(kl_div.item())
    return kl_divs, high_kl_features

kl_divergences_forward, high_kl_features_forward = compute_kl_for_features(pretrained_features_text, fine_tuned_features_text)
kl_divergences_reverse, high_kl_features_reverse = compute_kl_for_features(fine_tuned_features_text, pretrained_features_text)
high_kl_features_combined = list(set(high_kl_features_forward + high_kl_features_reverse))
high_kl_features_combined.sort()

print(f"{Fore.LIGHTGREEN_EX}\n{imagename} - High KL features (text): {', '.join(map(str, high_kl_features_combined))}{Style.RESET_ALL}")
with open(f"results/{imagename}-duo-high-KL-features-text.txt", "w", encoding='utf-8') as f:
    f.write(f"{', '.join(map(str, high_kl_features_combined))}")
    
# KL Divergence between activations of pre-trained and fine-tuned - vision threshold 0.5
def compute_kl_for_features(pretrained, fine_tuned, threshold=0.5):
    kl_divs = []
    high_kl_features = []
    for i in range(pretrained.shape[1]):
        p = pretrained[:, i]
        q = fine_tuned[:, i]
        kl_div = kl_divergence(F.softmax(p, dim=0), F.softmax(q, dim=0))
        if kl_div.item() >= threshold:  # Check if KL divergence exceeds the threshold
            high_kl_features.append(i)
        kl_divs.append(kl_div.item())
    return kl_divs, high_kl_features


kl_divergences_forward, high_kl_features_forward = compute_kl_for_features(pretrained_features_vision, fine_tuned_features_vision)
kl_divergences_reverse, high_kl_features_reverse = compute_kl_for_features(fine_tuned_features_vision, pretrained_features_vision)
high_kl_features_combined = list(set(high_kl_features_forward + high_kl_features_reverse))
high_kl_features_combined.sort()

print(f"{Fore.LIGHTYELLOW_EX}\n{imagename} - High KL features (vision): {', '.join(map(str, high_kl_features_combined))}{Style.RESET_ALL}")
with open(f"results/{imagename}-duo-high-KL-features-vision.txt", "w", encoding='utf-8') as f:
    f.write(f"{', '.join(map(str, high_kl_features_combined))}")
    
print("\nOutput saved to the 'results' folder.")