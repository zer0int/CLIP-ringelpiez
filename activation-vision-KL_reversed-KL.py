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

warnings.filterwarnings('ignore') # Eliminate spam about future pickle warning with latest torch
init(autoreset=True) # colorama

results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# Load models (pre-trained and fine-tuned)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model, preprocess = clip.load("ViT-L/14", device=device)
fine_tuned_model, _ = clip.load("ViT-L/14", device=device)
fine_tuned_model.load_state_dict(torch.load("finetune/ViT-L-14-BEST-smooth-GmP-ft-state_dict.pt", map_location="cuda"))

pretrained_model = pretrained_model.float()
fine_tuned_model = fine_tuned_model.float()

# Neuron capture hook class
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

# Function to register hooks on QuickGELU layers
def register_hooks(model, num_layers):
    hooks = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, QuickGELU):
            hook = ClipNeuronCaptureHook(module, layer_idx)
            hooks.append(hook)
            layer_idx += 1
            if layer_idx >= num_layers:
                break
    return hooks

# Function to run the model and collect activations
def run_model_and_collect_activations(model, image, hooks):
    # Run the model forward pass
    _ = model.encode_image(image)  # Forward pass
    # Collect activations for all hooked layers
    activations = [hook.get_activations() for hook in hooks]
    return activations

# Function to compute KL Divergence between two distributions
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

# KL Divergence between fine-tuned and pre-trained
def compute_kl_reverse(pretrained, fine_tuned):
    kl_divs = []
    for i in range(pretrained.shape[1]):  # Iterate over 4096 features
        p = fine_tuned[:, i]  # Fine-tuned activations for feature i
        q = pretrained[:, i]  # Pre-trained activations for feature i
        kl_div = kl_divergence(F.softmax(p, dim=0), F.softmax(q, dim=0))
        kl_divs.append(kl_div.item())
    return kl_divs


# KL Divergence between activations of pre-trained and fine-tuned
def compute_kl_for_features_list(pretrained, fine_tuned, threshold=0.5):
    kl_divs = []
    high_kl_features = []  # List to store high KL features
    for i in range(pretrained.shape[1]):  # Iterate over 4096 features
        p = pretrained[:, i]  # Pre-trained activations for feature i
        q = fine_tuned[:, i]  # Fine-tuned activations for feature i
        kl_div = kl_divergence(F.softmax(p, dim=0), F.softmax(q, dim=0))
        if kl_div.item() >= threshold:  # Check if KL divergence exceeds the threshold
            high_kl_features.append(i)
        kl_divs.append(kl_div.item())
    return kl_divs, high_kl_features

# Perform PCA on the activations
def perform_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance


# Plot PCA results
def plot_pca(pca_result_1, pca_result_2, var_1, var_2, imagename):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(pca_result_1[:, 0], pca_result_1[:, 1], color='blue', label='Pre-trained activations', alpha=0.5)
    plt.scatter(pca_result_2[:, 0], pca_result_2[:, 1], color='green', label='Fine-tuned activations', alpha=0.5)

    plt.title(f"{imagename} PCA -- Variance: Pre-trained={var_1.sum():.2f}, Fine-tuned={var_2.sum():.2f})")
    plt.xlabel(f"Principal Component 1 ({var_1[0]:.2%} variance)")
    plt.ylabel(f"Principal Component 2 ({var_1[1]:.2%} variance)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/VIS-{imagename}-PCA_CLIP.png")
    plt.close()

def color_divs(kl_divs):
    if kl_divs <= 1.00:
        return f"{Fore.LIGHTGREEN_EX}{kl_divs:.4f}{Style.RESET_ALL}"
    elif kl_divs <= 2.00:
        return f"{Fore.LIGHTYELLOW_EX}{kl_divs:.4f}{Style.RESET_ALL}"
    else:
        return f"{Fore.LIGHTRED_EX}{kl_divs:.4f}{Style.RESET_ALL}"


# Register hooks for both pre-trained and fine-tuned models
num_layers = 24  # Adjust if you're targeting specific layers
layer_idx = 22  # Index for Layer 22


pretrained_hooks = register_hooks(pretrained_model, num_layers)
fine_tuned_hooks = register_hooks(fine_tuned_model, num_layers)

# Load and preprocess the image
image_path = "images/bwcat_dog.png"
imagename = os.path.splitext(os.path.basename(image_path))[0]
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Run the models and capture activations for all post-GELU neurons
pretrained_act = run_model_and_collect_activations(pretrained_model, image, pretrained_hooks)
fine_tuned_act = run_model_and_collect_activations(fine_tuned_model, image, fine_tuned_hooks)

pretrained_features = torch.tensor(pretrained_act[layer_idx]).squeeze(1)  # Shape [257, 4096]
fine_tuned_features = torch.tensor(fine_tuned_act[layer_idx]).squeeze(1)  # Shape [257, 4096]

# Convert to numpy arrays for PCA
pretrained_np = pretrained_features.cpu().numpy()
fine_tuned_np = fine_tuned_features.cpu().numpy()

# PCA on pre-trained features
pca_pretrained, var_pretrained = perform_pca(pretrained_np)
# PCA on fine-tuned features
pca_finetuned, var_finetuned = perform_pca(fine_tuned_np)

# Plot PCA of pre-trained vs fine-tuned
plot_pca(pca_pretrained, pca_finetuned, var_pretrained, var_finetuned, imagename)

# Compute KL Divergence for Layer 22 activations
kl_divergences = compute_kl_for_features(pretrained_features, fine_tuned_features)

print('\n')
# Analyze and print KL divergence for all features with KL >= 0.5
for idx, kl_div in enumerate(kl_divergences):
    if kl_div >= 0.5:
        print(f"Feature Index {idx} - KL Divergence: {color_divs(kl_div)}")

mean_kl = np.mean(kl_divergences)
max_kl = np.max(kl_divergences)
min_kl = np.min(kl_divergences)

print(f"{Fore.LIGHTYELLOW_EX}Mean KL Divergence: {mean_kl:.4f}{Style.RESET_ALL}")
print(f"{Fore.LIGHTRED_EX}Max KL Divergence: {max_kl:.4f}{Style.RESET_ALL}")
print(f"{Fore.LIGHTGREEN_EX}Min KL Divergence: {min_kl:.4f}\n\n{Style.RESET_ALL}")

# Plot the KL Divergence for each feature
plt.figure(figsize=(10, 6))
plt.plot(kl_divergences, label="KL Divergence per Feature")
plt.title(f"{imagename} KL Div -- Pre-trained vs Fine-tuned Features (Layer {layer_idx})")
plt.xlabel("Feature Index")
plt.ylabel("KL Divergence")
plt.legend()
plt.grid(True)
plt.savefig(f"results/VIS-{imagename}-KL_CLIP.png")
plt.close()

# Compute KL(fine-tuned || pre-trained)
kl_divergences_reverse = compute_kl_reverse(pretrained_features, fine_tuned_features)

# Analyze and print KL divergence for all features with KL >= 0.5
for idx, kl_div in enumerate(kl_divergences_reverse):
    if kl_div >= 0.5:
        print(f"Feature Index {idx} - KL Divergence (reverse): {color_divs(kl_div)}")

# Analyze KL divergence in the reverse direction
mean_kl_reverse = np.mean(kl_divergences_reverse)
max_kl_reverse = np.max(kl_divergences_reverse)
min_kl_reverse = np.min(kl_divergences_reverse)

print(f"{Fore.LIGHTYELLOW_EX}Mean KL Divergence (Reverse): {mean_kl_reverse:.4f}{Style.RESET_ALL}")
print(f"{Fore.LIGHTRED_EX}Max KL Divergence (Reverse): {max_kl_reverse:.4f}{Style.RESET_ALL}")
print(f"{Fore.LIGHTGREEN_EX}Min KL Divergence (Reverse): {min_kl_reverse:.4f}{Style.RESET_ALL}")

# Plot the KL Divergence for each feature
plt.figure(figsize=(10, 6))
plt.plot(kl_divergences_reverse, label="KL Divergence per Feature")
plt.title(f"{imagename} KL Div -- Fine-tuned vs Pre-Trained Feautres (Layer {layer_idx})")
plt.xlabel("Feature Index")
plt.ylabel("KL Divergence")
plt.legend()
plt.grid(True)
plt.savefig(f"results/VIS-{imagename}-KL_rev_CLIP.png")
plt.close()

# Compute KL Divergence (forward)
kl_divergences_forward, high_kl_features_forward = compute_kl_for_features_list(pretrained_features, fine_tuned_features)

# Compute KL Divergence (reverse)
kl_divergences_reverse, high_kl_features_reverse = compute_kl_for_features_list(fine_tuned_features, pretrained_features)

# Combine the two lists and remove duplicates
high_kl_features_combined = list(set(high_kl_features_forward + high_kl_features_reverse))

# Sort the combined list for easier reading
high_kl_features_combined.sort()

# Print the combined list of high KL features
print(f"\n\nHigh KL features (forward + reverse): {', '.join(map(str, high_kl_features_combined))}\n")
with open(f"results/{imagename}-high-KL-features.txt", "w", encoding='utf-8') as f:
    f.write(f"{', '.join(map(str, high_kl_features_combined))}")
    

print("Output saved to the 'results' folder.")