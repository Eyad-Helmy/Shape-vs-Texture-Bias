# loads pretrained models and performs forward passes to get inferences from each model

import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

MODEL_REGISTERY = {
    "vgg16":    lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
    "resnet50":    lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    "alexnet":    lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
}

def load_all_models(model_names: list, device: str) -> dict:
    """
    Load pretrained models and move them to the target device.
 
    Arguments:
        model_names : list of strings — must be keys in MODEL_REGISTRY
        device      : 'cuda' or 'cpu'
 
    Returns:
        dict mapping model_name -> model (in eval mode, on device)
    """

    loaded_models = {}

    for name in model_names:
        if name not in MODEL_REGISTERY:
            raise ValueError(
                f"Unknown model {name}."
                f"Avaliable models are: {MODEL_REGISTERY.keys()}."
            )
        
        print(f"    loading {name}...")
        model = MODEL_REGISTERY[name]() # call the lambda inside it that calls the pre-trained model
        model.eval()
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters : {n_params:,}")
        print(f"    Device     : {device}")
 
        loaded_models[name] = model

    return loaded_models
        

@torch.no_grad()
def run_inference(model, input_tensor: torch.Tensor, device: str) -> "np.ndarray":
    """
    Run one forward pass through the model and return softmax probabilities.
 
    Arguments:
        model        : a loaded PyTorch model (output of load_all_models)
        input_tensor : (1, 3, 224, 224) normalized tensor
        device       : 'cuda' or 'cpu'
 
    Returns:
        (1000,) float32 numpy array — softmax probabilities for each class
    """

    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0]



# Tests claude
# # test model loads
# print("  [test_model_loads]")
 
# models_dict = load_all_models(['vgg16', 'resnet50'], device='cpu')
 
# assert isinstance(models_dict, dict), \
#     "load_all_models should return a dict"
# assert 'vgg16' in models_dict, \
#     "vgg16 key missing from returned dict"
# assert 'resnet50' in models_dict, \
#     "resnet50 key missing from returned dict"
# assert isinstance(models_dict['vgg16'], torch.nn.Module), \
#     "vgg16 value should be a torch.nn.Module"
# assert isinstance(models_dict['resnet50'], torch.nn.Module), \
#     "resnet50 value should be a torch.nn.Module"
 
# print("PASSED")

# # test evaluation mode
# for name, model in models_dict.items():
#     assert not model.training, \
#         f"{name} is still in training mode, .eval() was not called."
# print("PASSED")

# # test infrence output shape
# models_dict = load_all_models(['vgg16'], device='cpu')
# model = models_dict['vgg16']
 
# # torch.randn simulates a normalized image tensor (mean~0, std~1)
# # Same shape the real pipeline produces: (1, 3, 224, 224)
# dummy_input = torch.randn(1, 3, 224, 224)
 
# probs = run_inference(model, dummy_input, device='cpu')
 
# assert isinstance(probs, np.ndarray), \
#     f"Expected numpy array, got {type(probs)}"
# assert probs.shape == (1000,), \
#     f"Expected shape (1000,), got {probs.shape}"
# assert probs.dtype == np.float32, \
#     f"Expected float32, got {probs.dtype}"
 
# print("PASSED")

# # test inference output probablities range
# model = models_dict['resnet50']
 
# dummy_input = torch.randn(1, 3, 224, 224)
# probs = run_inference(model, dummy_input, device='cpu')
 
# assert probs.min() >= 0.0, \
#     f"Probabilities should be >= 0, found min={probs.min():.6f}"
# assert probs.max() <= 1.0, \
#     f"Probabilities should be <= 1, found max={probs.max():.6f}"
# assert abs(probs.sum() - 1.0) < 1e-4, \
#     f"Probabilities should sum to 1.0, got {probs.sum():.6f}"
 
# print("PASSED")\

# # test model is determenistic
# model = models_dict['vgg16']
 
# torch.manual_seed(0)
# dummy_input = torch.randn(1, 3, 224, 224)
 
# probs1 = run_inference(model, dummy_input, device='cpu')
# probs2 = run_inference(model, dummy_input, device='cpu')
 
# assert np.allclose(probs1, probs2, atol=1e-6), \
#     "Two identical inputs gave different outputs — model may be in train mode"
# assert probs1.argmax() == probs2.argmax(), \
#     f"Top-1 class changed between runs: {probs1.argmax()} vs {probs2.argmax()}"
 
# print("PASSED")

# # test single model load
# models_dict = load_all_models(['resnet50'], device='cpu')
 
# assert len(models_dict) == 1, \
#     f"Expected 1 model, got {len(models_dict)}"
# assert 'resnet50' in models_dict, \
#     "resnet50 should be in the dict"
# assert 'vgg16' not in models_dict, \
#     "vgg16 should NOT be in the dict when not requested"
 
# print("PASSED")