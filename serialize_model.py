#%%
import torch
from glomeruli_segmentation.model.unet import UNet

MODEL_PATH = 'hacking_kidney_16934_best_metric.model-384e1332.pth'
NEW_MODEL_PATH = "glomeruli_segmentation_16934_best_metric.model-384e1332.pth"
model = torch.load(MODEL_PATH, map_location="cpu")

arch_split = model["args"][0].split("_")
decoder = arch_split[1] if len(arch_split) > 1 else "simple"
image_classification = arch_split[2] == "c" if len(arch_split) > 2 else False
model_kwargs: dict = model["kwargs"]

new_model_kwargs = dict(
    decoder=arch_split[1] if len(arch_split) > 1 else "simple",
    backbone=model["args"][1],
    num_head_features=16 * model_kwargs["n_classes"],
    cat_features=True,
)
del model['kwargs']['activation']
del model['kwargs']['frozen_layers']
del model['kwargs']['feature_layers']
del model['kwargs']['frozen_batchnorm']

model["kwargs"].update(new_model_kwargs)
del model["args"]
torch.save(model, NEW_MODEL_PATH)
#%%
new_model = torch.load(NEW_MODEL_PATH)
print(new_model["kwargs"], len(new_model['state_dict']))
#%%
unet = UNet(**new_model["kwargs"])
print(unet)
#%%
unet.load_state_dict(new_model['state_dict'])
print(unet)
