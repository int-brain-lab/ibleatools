import os

from spatial_encoder_model import *
from spatial_encoder_utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
train_model = True

# 0) Load relevant data from AWS
# Download pretrained model from aws
local_path = os.getcwd()
model_name = 'SE_model'
path_to_model = download_pretrained_model(local_path, model_name, overwrite=True)

# 1) Build a context manager - enable drawing the agea & merfish PC vector given xyz coordinates
print("Generating context grid")
cfg = AtlasPCAConfig()
ctx_manager = ContextAtlasManager(cfg, model_name, regenerate_context=False)

# 2) Load the ephys atlas probes - excluding the misaligned probes
print("Loading insertion data")
pid_str, ephys, probe_positions, probe_planned_positions = LoadInsertionData()

# 3) Build data loaders for model training. The loaders have a special collate function that
#    samples the nearest neighbors from the training data
M_MAX = 8
RADIUS_UM = 500

train_loader, val_loader, test_loader, e_mean, e_std, ctx_mean, ctx_std = build_channels_plus_emptyvoxels_with_neighbors(
    ctx_manager=ctx_manager,
    ephys=ephys,
    probe_positions=probe_positions,
    RADIUS_UM=RADIUS_UM,
    M_MAX=M_MAX)

# 4) Instantiate the spatial encoder model. Inputs: context vector + ephys of NN. output: ephys at position xyz
F_ctx = cfg.n_cell_pcs + cfg.n_gene_pcs
F_e = ephys.shape[-1]
F_REG = 0
heteroscedastic = False

model = NeighborInpaintingModel(
    f_ctx=F_ctx, f_ephys=F_e, f_out=F_e, f_region=F_REG, e_mean=e_mean, e_std=e_std, ctx_mean=ctx_mean, ctx_std=ctx_std,
    d_model=128, nhead=8, depth=2, neighbor_self_attn=False, heteroscedastic=heteroscedastic, drop=0.15
).to(device)

# 5) Train the model
if(train_model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    model, meters, best_ep, best_val = train_hybrid(
        model, train_loader, val_loader, opt,
        epochs=100, device=device,
        lambda_sup=0.05, lambda_ctr=1.0, tau=0.05, pos_radius_um=RADIUS_UM,
        heteroscedastic=heteroscedastic, early_stopping=True, patience=2, min_delta=5e-4, ephys_drop=0.0,
        monitor="val/sup", mode="min",
        checkpoint_path="best_model.pt",
        lr_scheduler=sched
    )
    torch.save(model.state_dict(), f'{model_name}/SE_model.pth')
else:
    model.load_state_dict(torch.load(path_to_model))

e_mean = model.e_mean
e_std = model.e_std

# 6) Evaluate R² per feature (original scale) on the test set
r2_per_feat = evaluate_r2_per_feature(model, test_loader, e_mean.to(device), e_std.to(device), device=device)

print("Test R^2 per feature (len={}):".format(len(r2_per_feat)))
print(r2_per_feat.numpy())
print("Mean R^2:", float(r2_per_feat.mean().item()))