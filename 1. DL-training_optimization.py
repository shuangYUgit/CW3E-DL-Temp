
# =====================================================
# IMPORTS
# =====================================================
import os
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from tqdm import tqdm
import copy


# =====================================================
# CONFIG
# =====================================================
v = 'v8.5'
exp = '_exp_0_'

batch_size = 20000
epochs = 250
learning_rate = 1e-3

save_dir = "/cw3e/mead/projects/cwp190/projects/post-processing/output/" + v + "/models"
os.makedirs(save_dir, exist_ok=True)

lead_time = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96,
             102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168]
lead_time_index = 3   # -> 24h

data_dir = ('/cw3e/mead/projects/cwp190/projects/post-processing/data/features/'
            'lead_time_' + str(lead_time[lead_time_index]) + '/')


# =====================================================
# LOSS FUNCTION -- weighted pinball loss with a median-band enhancement
# =====================================================
def pinball_loss(y_pred: torch.Tensor,
                  y_true: torch.Tensor,
                  taus: torch.Tensor,
                  q_weights: torch.Tensor | None = None,
                  lambda_mid: float = 0.8,   # weight of the median-band loss term (tunable)
                  sigma_mid: float = 0.02    # Gaussian width of the median band (smaller = narrower)
                  ) -> torch.Tensor:
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(1)                   # (batch,1)
    err = y_true - y_pred                               # (batch, Q)
    taus = taus.reshape(1, -1).to(y_pred.device)
    loss_per_q = torch.maximum(taus * err, (taus - 1.0) * err)  # (batch, Q)

    # -- main term: standard (optionally weighted) pinball loss --
    if q_weights is None:
        base_loss = loss_per_q.mean()
    else:
        w = q_weights.reshape(1, -1).to(y_pred.device)          # (1, Q)
        w = w * (w.numel() / (w.sum() + 1e-8))                  # normalize to mean = 1
        base_loss = (loss_per_q * w).mean()

    # -- additional term: narrow Gaussian bump around the median --
    # lambda_mid=0 disables this term; sigma_mid controls its width
    # (smaller = more tightly focused on the median).
    if lambda_mid > 0 and sigma_mid > 0:
        tvec = taus.squeeze(0)                                  # (Q,)
        bump = torch.exp(-0.5 * ((tvec - 0.5) / sigma_mid) ** 2).reshape(1, -1)
        bump = bump.to(y_pred.device)
        bump = bump * (bump.numel() / (bump.sum() + 1e-8))      # normalize to mean = 1
        midband_loss = (loss_per_q * bump).mean()
        loss = base_loss + lambda_mid * midband_loss
    else:
        loss = base_loss

    return loss


# =====================================================
# DATA LOADING
# =====================================================
all_files = sorted([
    f for f in os.listdir(data_dir)
    if f.startswith("wrf_obs_")
       and f.endswith(".npz")
       and "_predict_" in f
])

features_list = []
wrf_predict_time_list = []
obs_in_predict_time_list = []

for fname in all_files:
    fpath = os.path.join(data_dir, fname)
    npz = np.load(fpath, allow_pickle=True)  # allow_pickle=True in case they're objects/datetimes
    features = npz["feature"][:, 0:-1]
    wrf_predict_time = npz["wrf_predict_time"]
    obs_in_predict_time = npz["obs_in_predict_time"]
    features_list.append(features)
    wrf_predict_time_list.append(wrf_predict_time)
    obs_in_predict_time_list.append(obs_in_predict_time)

len(features_list)
len(wrf_predict_time_list)
len(obs_in_predict_time_list)


# =====================================================
# PREPROCESSING -- stack into (days, basins, features), min-max scale
# =====================================================
X = np.stack(features_list, axis=0)
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = X_reshaped.copy().astype(float)
fmin = X_scaled.min(axis=0)   # (10,)
fmax = X_scaled.max(axis=0)   # (10,)
X_scaled = (X_scaled - fmin) / (fmax - fmin)
X_scaled[:, 6] = X_reshaped[:, 6]   # column 6 is the categorical station_id -- leave unscaled
X_recovered = X_scaled.reshape(449, 527, 10)
Y = np.stack(obs_in_predict_time_list, axis=0)

n_days = X_recovered.shape[0]
n_basins = X_recovered.shape[1]

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


# =====================================================
# TRAIN / VAL SPLIT
# =====================================================
pick_day = 448

test_days = pick_day
train_days_pre = np.setdiff1d(np.arange(n_days), test_days)
train_days = np.random.choice(train_days_pre, size=400, replace=False)
val_days = np.setdiff1d(train_days_pre, train_days)

x_train = X_recovered[train_days, :, :].reshape(-1, 10)
y_train = Y[train_days, :].reshape(-1,) - 273.15

x_val = X_recovered[val_days, :, :].reshape(-1, 10)
y_val = Y[val_days, :].reshape(-1,) - 273.15

train_x = torch.tensor(x_train, dtype=torch.float32)
val_x = torch.tensor(x_val, dtype=torch.float32)

train_y = torch.tensor(y_train, dtype=torch.float32)
val_y = torch.tensor(y_val, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================
# MODEL -- median-anchored QRNN (monotonic, non-crossing quantiles)
# =====================================================
class QRNNEmbedModelMedianAnchored(nn.Module):
    def __init__(self,
                 taus,                  # 1D tensor/list of quantile levels, ascending e.g. [0.01,...,0.5,...,0.99]
                 n_features=10,         # total feature dim going into the net
                 embedding_dim=527,     # number of unique stations/basins
                 embedding_size=5,      # size of the station embedding
                 hidden_nodes=[10, 10],
                 activation="softplus"):
        super().__init__()
        # -- store / analyze taus --
        taus = torch.as_tensor(taus, dtype=torch.float32)
        assert torch.all(taus[:-1] <= taus[1:]), "taus must be sorted ascending"
        self.register_buffer("taus", taus)  # so it's moved to cuda with model
        # find index of "median" quantile = tau closest to 0.5
        with torch.no_grad():
            mid_idx = torch.argmin(torch.abs(taus - 0.5)).item()
        self.mid_idx = mid_idx
        # split counts
        self.K_lower = mid_idx                  # number of taus strictly below the mid quantile
        self.K_upper = len(taus) - mid_idx - 1   # number above
        # sanity
        assert self.K_lower >= 0
        assert self.K_upper >= 0
        assert self.K_lower + 1 + self.K_upper == len(taus)

        # station/basin embedding (feature index 6)
        self.loc_embedding = nn.Embedding(embedding_dim + 1, embedding_size)

        if activation == "softplus":
            self.hidden_act = nn.Softplus()
        else:
            self.hidden_act = nn.ReLU()

        input_dim = (n_features - 1) + embedding_size  # 10-1 + emb
        layers = []
        cur_dim = input_dim
        for h in hidden_nodes:
            layers.append(nn.Linear(cur_dim, h))
            layers.append(self.hidden_act)
            cur_dim = h
        self.trunk = nn.Sequential(*layers)

        self.mid_head = nn.Linear(cur_dim, 1)
        # if there are no lower/upper taus, make a dummy 0-len layer
        self.lower_head = nn.Linear(cur_dim, self.K_lower) if self.K_lower > 0 else None
        self.upper_head = nn.Linear(cur_dim, self.K_upper) if self.K_upper > 0 else None
        self.pos_fn = nn.Softplus()  # strictly positive-ish

    def forward(self, x):
        """
        x shape: (batch, n_features)
        Index 6 is station_id (categorical).
        """
        # 1) station embedding
        station_id = x[:, 6].long()  # (batch,)
        loc_emb = self.loc_embedding(station_id)  # (batch, embedding_size)
        # 2) numeric features except station_id: concat [0..5] and [7..end]
        numerical = torch.cat([x[:, :6], x[:, 7:]], dim=1)  # (batch, 9)
        # 3) combine numeric + embedding
        feats = torch.cat([numerical, loc_emb], dim=1)       # (batch, 9+emb)
        # 4) trunk MLP
        h = self.trunk(feats)                                # (batch, hidden_last)
        # 5) mid quantile
        q_mid = self.mid_head(h)                             # (batch, 1)
        # 6) lower side (taus < median)
        if self.K_lower > 0:
            raw_down = self.lower_head(h)                   # (batch, K_lower)
            down_steps = self.pos_fn(raw_down)              # >=0
            down_cum = torch.cumsum(down_steps, dim=1)      # (batch, K_lower)
            # closest-to-mid quantile is q_mid - down_cum[:,0]
            q_lower_desc = q_mid - down_cum                 # (batch, K_lower), strictly decreasing
            # taus are ascending (smallest tau first), so flip to align
            # lowest tau first (most subtracted)
            q_lower = torch.flip(q_lower_desc, dims=[1])    # (batch, K_lower), nondecreasing left->right
        else:
            q_lower = torch.empty((x.shape[0], 0), device=x.device, dtype=q_mid.dtype)
        # 7) upper side (taus > median)
        if self.K_upper > 0:
            raw_up = self.upper_head(h)                     # (batch, K_upper)
            up_steps = self.pos_fn(raw_up)                  # >=0
            up_cum = torch.cumsum(up_steps, dim=1)          # (batch, K_upper)
            q_upper = q_mid + up_cum                        # (batch, K_upper), strictly increasing
        else:
            q_upper = torch.empty((x.shape[0], 0), device=x.device, dtype=q_mid.dtype)
        # 8) stitch full quantile fan in ascending tau order:
        #    taus[0:mid_idx]  -> q_lower
        #    taus[mid_idx]    -> q_mid
        #    taus[mid_idx+1:] -> q_upper
        quantiles_out = torch.cat([q_lower, q_mid, q_upper], dim=1)  # (batch, len(taus))
        return quantiles_out


# =====================================================
# MODEL INSTANTIATION
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_quantiles = 44
taus = torch.linspace(0.001, 0.999, steps=num_quantiles).to(device)

model = QRNNEmbedModelMedianAnchored(
    taus=taus,
    n_features=10,
    embedding_dim=527,      # number of basins
    embedding_size=5,
    hidden_nodes=[10, 10],
    activation="softplus"
).to(device)

# Alternate quantile-weighting scheme (kept for reference, not used):
#   with torch.no_grad():
#       eps_w = 0.2                                # keep a positive floor at ends
#       q_weights = eps_w + torch.sin(math.pi * taus)  # shape (Q,)

with torch.no_grad():
    eps_w = 0.1
    amp = 1.5
    q_weights = eps_w + amp * (torch.sin(math.pi * taus) ** 1.5)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
summary(model, input_size=(10,))


# =====================================================
# TRAINING (validation-based early stopping; save best weights only)
# =====================================================
patience_es = 10
min_improve = 1e-3
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                               patience=5, min_lr=1e-5, verbose=True)

best_val = float('inf')
best_epoch = -1
epochs_no_improve = 0
best_state_dict = None
save_epoch = None

# Column names kept as ["epoch", "train_loss", "test_loss"] for
# downstream compatibility with existing plotting/analysis scripts.
loss_train_test = []

for epoch in range(1, epochs + 1):
    # ---------------- train ----------------
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch:03d}")
    for i, (x_batch, y_batch) in progress_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = pinball_loss(y_pred, y_batch, taus, q_weights=q_weights)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)
    avg_train_loss = train_loss / len(train_loader.dataset)

    # ---------------- validation ----------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = pinball_loss(y_pred, y_batch, taus)  # unweighted for validation
            val_loss += loss.item() * x_batch.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)

    # ---------------- log ----------------
    loss_train_test.append([epoch, avg_train_loss, avg_val_loss])
    loss_df = pd.DataFrame(loss_train_test, columns=["epoch", "train_loss", "test_loss"])
    csv_path = os.path.join(save_dir, v + exp + "train_val_loss_pick_" + str(pick_day) + ".csv")
    loss_df.to_csv(csv_path, index=False)
    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # ---------------- LR scheduler ----------------
    scheduler.step(avg_val_loss)

    # ---------------- early stopping bookkeeping ----------------
    if avg_val_loss < best_val - min_improve:
        best_val = avg_val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        best_state_dict = copy.deepcopy(model.state_dict())   # keep best weights in memory
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience_es:   # trigger early stop
        print(f"[Early Stop] No improvement for {patience_es} epochs. "
              f"Best epoch = {best_epoch}, best val = {best_val:.6f}")
        save_epoch = best_epoch
        # save the best weights exactly once, here
        final_path = os.path.join(save_dir, f"qrnn_epoch{save_epoch:03d}" + v + exp + "pick_" + str(pick_day) + ".pt")
        torch.save(best_state_dict, final_path)
        break

# ---------------- wrap-up after the loop (save last epoch if no early stop) ----------------
if save_epoch is None:   # no early stop -- ran the full epoch budget
    save_epoch = epochs
    final_path = os.path.join(save_dir, f"qrnn_epoch{save_epoch:03d}" + v + exp + ".pt")
    torch.save(model.state_dict(), final_path)
    print(f"[Training Finished] Reached max epochs = {epochs}. "
          f"Final saved: {final_path}")
else:
    print(f"[Training Finished] Early-stopped at best epoch = {save_epoch}. "
          f"Final saved: {final_path}")

print(f"--> save_epoch = {save_epoch}")   # explicit print for reference when reloading later
