import os
import torch
import torch.nn.functional as F
import clip
from clip.model import QuickGELU
from PIL import Image
from colorama import Fore, Style, init
import warnings

warnings.filterwarnings('ignore') # Eliminate spam about future pickle warning with latest torch
init(autoreset=True) # colorama

results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

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

def detect_num_layers_and_default_layers(model):
    num_layers = 0

    # vision transformer
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)

    return num_layers

def run_model_and_collect_activations(model, image, hooks):
    # Run the model forward pass
    _ = model.encode_image(image)  # Forward pass
    # Collect activations for all hooked layers
    activations = [hook.get_activations() for hook in hooks]
    return activations
    
def color_cosine_similarity(cos_sim):
    if cos_sim >= 0.99:
        return f"{Fore.LIGHTGREEN_EX}{cos_sim:.4f}{Style.RESET_ALL}"
    elif cos_sim >= 0.70:
        return f"{Fore.LIGHTYELLOW_EX}{cos_sim:.4f}{Style.RESET_ALL}"
    else:
        return f"{Fore.LIGHTRED_EX}{cos_sim:.4f}{Style.RESET_ALL}"


# Load models (pre-trained and fine-tuned)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model, preprocess = clip.load("ViT-L/14", device=device)
fine_tuned_model, _ = clip.load("ViT-L/14", device=device)
fine_tuned_model.load_state_dict(torch.load("finetune/ViT-L-14-BEST-smooth-GmP-ft-state_dict.pt", map_location="cuda"))

# Automatically detect the number of layers and the default layer indices to ensure compatibility with all models
num_layers = detect_num_layers_and_default_layers(pretrained_model)
print(f"\nDetected {num_layers} layers in the vision transformer.")

pretrained_hooks = register_hooks(pretrained_model, num_layers)
fine_tuned_hooks = register_hooks(fine_tuned_model, num_layers)

# Load and preprocess the image
image_path = "images/bwcat_dog.png"
imagename = os.path.splitext(os.path.basename(image_path))[0]
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

pretrained_activations = run_model_and_collect_activations(pretrained_model, image, pretrained_hooks)
fine_tuned_activations = run_model_and_collect_activations(fine_tuned_model, image, fine_tuned_hooks)

print(f"\n{Fore.LIGHTYELLOW_EX}Getting similarities for {imagename}...\n")

for idx, (pretrained_act, fine_tuned_act) in enumerate(zip(pretrained_activations, fine_tuned_activations)):
    if pretrained_act is not None and fine_tuned_act is not None:
        #print(f"Pre-trained Layer {idx} activations shape: {pretrained_act.shape}")
        #print(f"Fine-tuned Layer {idx} activations shape: {fine_tuned_act.shape}")

        # Selecting only the [CLS] token for comparison
        pretrained_cls_token = pretrained_act[0, 0, :]  # First token (CLS)
        fine_tuned_cls_token = fine_tuned_act[0, 0, :]  # First token (CLS)

        # Compare neuron activations
        def cosine_similarity(vec1, vec2):
            vec1 = torch.tensor(vec1).float()
            vec2 = torch.tensor(vec2).float()
            vec1 = F.normalize(vec1, p=2, dim=-1)
            vec2 = F.normalize(vec2, p=2, dim=-1)
            return torch.matmul(vec1, vec2.T)

        # Compute cosine similarity between corresponding neurons
        cos_sim = cosine_similarity(pretrained_cls_token, fine_tuned_cls_token).item()
        # Color coding the cosine similarity values
        colored_sim = color_cosine_similarity(cos_sim)

        print(f"Cosine similarity for Layer {idx} [CLS] token: {colored_sim}")
        with open(f"results/{imagename}-CLS-Layer-cos_sim.txt", "a", encoding='utf-8') as f:
            f.write(f"Layer {idx} [CLS] token: {cos_sim}\n")
    else:
        print(f"Layer {idx} activations were not captured correctly.")
        
print("\nOutput saved to the 'results' folder.")