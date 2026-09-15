
# =====================================================
# MODULE 1: IMPORTS
# =====================================================
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# =====================================================
# MODULE 2: CONFIG
# =====================================================
v = 'v9.7'
v2 = 'v8.5.1'
exp = '_exp_1_'

lead_time = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96,
             102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168]

# =====================================================
# MODULE 3: PART 1 -- PER-BASIN-GROUP METRICS, ALL LEAD TIMES
# =====================================================
for lead_time_index, LT in enumerate(lead_time):
    out_dir = ("/cw3e/mead/projects/cwp190/projects/post-processing/output/" + v
               + "/DL_out/" + v2 + exp + "/basin/lead_" + str(lead_time[lead_time_index]) + "/")
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for basin_group in range(17):

        # ---- basin group setup ----
        # Stored purely as a label/reminder that the evaluated event below
        # is a symmetric +/- band (not used in any computation itself).
        band_half_width_degC = 2.5
        shp = "/cw3e/mead/projects/cwp190/projects/post-processing/data/CNRFC/basin_information/CNRFCbasins/CNRFC_Basins_Sub.shp"
        gdf = gpd.read_file(shp)
        gdf_unique = gdf.drop_duplicates(subset=["ForecastGr"], keep="first")["ForecastGr"].dropna().tolist()
        basin_group_name = gdf_unique[basin_group]
        names_in_group = gdf.loc[gdf["ForecastGr"] == basin_group_name, "Name"].tolist()
        npz_ex = np.load('/cw3e/mead/projects/cwp190/projects/post-processing/data/features/lead_time_24/wrf_obs_24_start_2025033100_predict_2025040100.npz', allow_pickle=True)
        basin_zone_name = npz_ex["basin_zone_name"]
        mask = np.isin(basin_zone_name, np.array(names_in_group, dtype=str))
        indices_all = np.where(mask)[0]
        matched_names = basin_zone_name[indices_all]

        # ---- load LODO predictions for this basin group ----
        csvname = glob.glob(f"/cw3e/mead/projects/cwp190/projects/post-processing/output/{v2}/lead_{lead_time[lead_time_index]}" + "h/figure_experimentv*")[0].split('_')[-3][5:]
        check = int(csvname)
        pred_np = np.zeros((check * (indices_all.shape[0]), 44))
        truth_np = np.zeros((check * (indices_all.shape[0]),))
        for i in range(check):
            pick = i
            pred_files = glob.glob("/cw3e/mead/projects/cwp190/projects/post-processing/output/" + v2 + "/lead_" + str(lead_time[lead_time_index]) + "h/qrnn_epoch*_test_pred" + v2 + "_exp_1_pick_" + str(pick) + "_lead_" + str(lead_time[lead_time_index]) + ".npy")[0]
            obs_files = glob.glob("/cw3e/mead/projects/cwp190/projects/post-processing/output/" + v2 + "/lead_" + str(lead_time[lead_time_index]) + "h/qrnn_epoch*_test_obs" + v2 + "_exp_1_pick_" + str(pick) + "_lead_" + str(lead_time[lead_time_index]) + ".npy")[0]
            pred_np[i * (indices_all.shape[0]):(i + 1) * (indices_all.shape[0]), :] = np.load(pred_files)[indices_all, :]
            truth_np[i * (indices_all.shape[0]):(i + 1) * (indices_all.shape[0]),] = np.load(obs_files)[indices_all,]
            print(pick)

        # ---- metric functions ----
        taus = np.linspace(0.001, 0.999, 44)  # default grid

        def _invert_tau_given_value(q_row, taus, y):
            """Piecewise-linear inverse of Q(tau) at value y -> tau* in [0,1]."""
            idx = np.searchsorted(q_row, y, side='right')
            if idx == 0:
                return 0.0
            if idx == len(q_row):
                return 1.0
            q_lo, q_hi = q_row[idx - 1], q_row[idx]
            tau_lo, tau_hi = taus[idx - 1], taus[idx]
            if q_hi == q_lo:
                return 0.5 * (tau_lo + tau_hi)
            w = (y - q_lo) / (q_hi - q_lo)
            return float(tau_lo + w * (tau_hi - tau_lo))

        def pit_from_quantiles(pred_q, taus, truth):
            """pred_q: (N, Q) or (...,Q); truth: (N,) or (...) -> flattened PIT."""
            q = np.asarray(pred_q); t = np.asarray(truth); taus = np.asarray(taus)
            N = int(np.prod(t.shape)); Q = q.shape[-1]
            q2 = q.reshape(N, Q); t2 = t.reshape(N)
            out = np.empty(N, dtype=float)
            for i in range(N):
                out[i] = _invert_tau_given_value(q2[i], taus, float(t2[i]))
            return np.clip(out.reshape(t.shape), 0.0, 1.0).ravel()

        def reliability_decomposition(probs, outs, K=44, right=True):
            """Return REL, RES, UNC, BS, BSS + curve points (only non-empty bins)."""
            probs = np.asarray(probs).ravel(); outs = np.asarray(outs).ravel()
            N = probs.size
            edges = np.linspace(0, 1, K + 1)
            idx = np.digitize(probs, edges, right=right) - 1
            idx = np.clip(idx, 0, K - 1)

            n_k = np.zeros(K, dtype=int)
            f_k = np.full(K, np.nan)
            o_k = np.full(K, np.nan)
            for k in range(K):
                m = (idx == k); n = int(m.sum()); n_k[k] = n
                if n > 0:
                    f_k[k] = probs[m].mean()
                    o_k[k] = outs[m].mean()

            valid = n_k > 0
            wk = n_k[valid] / N
            fkv, okv = f_k[valid], o_k[valid]
            o_bar = outs.mean()
            REL = np.sum(wk * (fkv - okv) ** 2)
            RES = np.sum(wk * (okv - o_bar) ** 2)
            UNC = o_bar * (1.0 - o_bar)
            BS = np.mean((probs - outs) ** 2)
            BSS = (1 - BS / UNC) if UNC > 0 else np.nan
            return dict(REL=REL, RES=RES, UNC=UNC, BS=BS, BSS=BSS, x=fkv, y=okv, n_k=n_k, edges=edges)

        def sharpness_q90_q10_quantile(pred_q, taus):
            """Interpolate to tau=0.10 and 0.90, then average width."""
            taus = np.asarray(taus)

            def interp_tau(q_mat, tau):
                idx = np.searchsorted(taus, tau, side="right")
                if idx == 0:
                    return q_mat[..., 0]
                if idx == len(taus):
                    return q_mat[..., -1]
                q_lo = q_mat[..., idx - 1]; q_hi = q_mat[..., idx]
                t_lo, t_hi = taus[idx - 1], taus[idx]
                w = (tau - t_lo) / (t_hi - t_lo)
                return q_lo + w * (q_hi - q_lo)
            q10 = interp_tau(pred_q, 0.10)
            q90 = interp_tau(pred_q, 0.90)
            return float(np.mean(q90 - q10))

        def crps_from_quantiles(pred_q, taus, truth):
            """CRPS = 2 * integral_0^1 rho_tau(y - Q(tau)) dtau (trapezoid over taus)."""
            q = np.asarray(pred_q); t = np.asarray(truth); taus = np.asarray(taus)
            u = t[..., None] - q
            rho = np.empty_like(q, dtype=float)
            for j, tau in enumerate(taus):
                rho[..., j] = u[..., j] * (tau - (u[..., j] < 0).astype(float))
            dtaus = np.diff(taus)
            trap = 0.5 * (rho[..., :-1] + rho[..., 1:]) * dtaus
            crps = 2.0 * np.sum(trap, axis=-1).ravel()
            return float(np.mean(crps)), crps

        def crps_climatology_from_obs(obs_all, y, subsample=5000, seed=0):
            """
            Approximate CRPS of climatology F_clim using the entire observations 'obs_all'.
            CRPS(F_clim, y) ~= mean|X - y| - 0.5 * E|X - X'|.
            """
            rng = np.random.default_rng(seed)
            obs = np.asarray(obs_all).ravel()
            if (subsample is not None) and (subsample < obs.size):
                idx = rng.choice(obs.size, size=subsample, replace=False)
                obs = obs[idx]

            obs_sorted = np.sort(obs)
            M = obs_sorted.size
            csum = np.cumsum(obs_sorted)
            idx = np.arange(M)
            contrib = idx * obs_sorted - np.concatenate(([0.0], csum[:-1]))
            pair = 2.0 * np.sum(contrib) / (M * M)
            const = 0.5 * pair

            y = np.asarray(y).ravel()
            B = 20000
            out = np.empty_like(y, dtype=float)
            for s in range(0, y.size, B):
                yy = y[s:s + B][:, None]
                out[s:s + B] = np.mean(np.abs(obs[None, :] - yy), axis=1) - const
            return out

        def crpss_from_crps(crps_f, crps_ref):
            """CRPSS = 1 - CRPS_forecast / CRPS_reference (element-wise)."""
            return 1.0 - (crps_f / crps_ref)

        def probs_from_quantiles_event_band(pred_q, taus, truth, low, high):
            """
            Event: low <= Y <= high
            p_hat = F(high) - F(low)
            o     = 1{low <= truth <= high}
            """
            q = np.asarray(pred_q); t = np.asarray(truth); taus = np.asarray(taus)
            N = int(np.prod(t.shape)); Q = q.shape[-1]
            q2 = q.reshape(N, Q); t2 = t.reshape(N)

            F_low = np.empty(N, dtype=float)
            F_high = np.empty(N, dtype=float)
            for i in range(N):
                F_low[i] = _invert_tau_given_value(q2[i], taus, low)
                F_high[i] = _invert_tau_given_value(q2[i], taus, high)

            p_hat = F_high - F_low
            p_hat = np.clip(p_hat, 0.0, 1.0)

            o = ((t2 >= low) & (t2 <= high)).astype(float)

            return p_hat.reshape(t.shape).ravel(), o.reshape(t.shape).ravel()

        def plot_panel(pit_vals, title_right, metrics, bins_pit=20, basin_group_name=basin_group_name):
            """
            Draw a 1x2 panel: left = PIT histogram; right = reliability diagram with metrics box.
            metrics must include keys: REL, RES, BS, BSS, SHARP, CRPSS, x, y
            """
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

            ax = axes[0]
            bins = np.linspace(0, 1, bins_pit + 1)
            ax.hist(pit_vals, bins=bins, edgecolor='black', alpha=0.85)
            ax.axhline(len(pit_vals) / bins_pit, linestyle=':', linewidth=1)
            ax.set_xlabel("u = F(y)")
            ax.set_ylabel("Count")
            ax.set_title("PIT Histogram")
            ax.text(0.98, 0.92, f"Mean PIT = {np.mean(pit_vals):.4f}", ha='right', va='top',
                    transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

            ax2 = axes[1]
            ax2.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='k', label='perfect reliability')
            ax2.scatter(metrics["x"], metrics["y"], s=18, alpha=0.85)
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            ax2.set_xlabel("Forecast probability")
            ax2.set_ylabel("Observed frequency")
            ax2.set_title(title_right)
            txt = (f"REL={metrics['REL']:.4f}\n"
                   f"RES={metrics['RES']:.4f}\n"
                   f"BS ={metrics['BS']:.4f}\n"
                   f"BSS={metrics['BSS']:.4f}\n"
                   f"Sharpness (Q90-Q10)={metrics.get('SHARP', np.nan):.2f}\n"
                   f"CRPSS={metrics.get('CRPSS', np.nan):.4f}")
            ax2.text(0.98, 0.05, txt, ha='right', va='bottom',
                     transform=ax2.transAxes, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='gray', alpha=0.9))
            plt.tight_layout()
            return fig, axes

        # ---- apply: quantile-based evaluation, event = -2.5 to 2.5 degC ----
        pits_q = pit_from_quantiles(pred_np, taus, truth_np)

        low, high = -2.5, 2.5
        probs_q, outs_q = probs_from_quantiles_event_band(pred_np, taus, truth_np, low, high)
        rel_q = reliability_decomposition(probs_q, outs_q, K=44, right=True)

        sharp_q = sharpness_q90_q10_quantile(pred_np, taus)

        mean_crps_q, crps_q_vec = crps_from_quantiles(pred_np, taus, truth_np)
        crps_ref_vec_q = crps_climatology_from_obs(truth_np, y=truth_np, subsample=5000, seed=0)
        crpss_q_vec = crpss_from_crps(crps_q_vec, crps_ref_vec_q[:crps_q_vec.size] if crps_ref_vec_q.size != crps_q_vec.size else crps_ref_vec_q)
        CRPSS_q = float(np.nanmean(crpss_q_vec))

        metrics_q = dict(REL=rel_q["REL"], RES=rel_q["RES"], BS=rel_q["BS"], BSS=rel_q["BSS"],
                          SHARP=sharp_q, CRPSS=CRPSS_q, x=rel_q["x"], y=rel_q["y"])
        title_q = f"Reliability Diagram(-2.5 - 2.5 degC), " + basin_group_name
        fig_q, axes_q = plot_panel(pits_q, title_right=title_q, metrics=metrics_q, bins_pit=20, basin_group_name=basin_group_name)

        plt.savefig(os.path.join(out_dir, f"quantile_{v2}{exp}lead{lead_time[lead_time_index]}_basin_{basin_group_name}_range_-2.5_to_2.5.png"), dpi=150)
        rows.append({
            'basin': basin_group_name,
            'lead_time': lead_time[lead_time_index],
            'mean_PIT': float(np.mean(pits_q)),
            'band_half_width_degC': float(band_half_width_degC),
            'REL': float(rel_q["REL"]),
            'RES': float(rel_q["RES"]),
            'BS': float(rel_q["BS"]),
            'BSS': float(rel_q["BSS"]),
            'sharpness': float(sharp_q),
            'CRPSS': float(CRPSS_q),
        })

    # ---- save per-lead-time CSV across all basin groups ----
    df = pd.DataFrame(rows)
    df = df.set_index('basin')
    cols = ['lead_time', 'mean_PIT', 'band_half_width_degC', 'REL', 'RES', 'BS', 'BSS', 'sharpness', 'CRPSS']
    df = df[cols]
    csv_path = os.path.join(out_dir, f"DL_quantile_metrics_lead{lead_time[lead_time_index]}_by_each_basin_range_-2.5_to_2.5.csv")
    df.to_csv(csv_path, index=True)

    print(f"[Done] Figures saved to: {out_dir}")
    print(f"[Done] Metrics CSV: {csv_path}")


# =====================================================
# MODULE 4: PART 2 -- DL vs WestWRF COMPARISON PLOTS
# =====================================================
out_dir = "/cw3e/mead/projects/cwp190/projects/post-processing/output/" + v + "/DL_out/" + v2 + exp + '/all_lead_dl_vs_westWRF/'
os.makedirs(out_dir, exist_ok=True)

metrics = ["mean_PIT", "REL", "RES", "BS", "BSS", "sharpness", "CRPSS"]

for basin_group in range(17):
    data_wrf = np.zeros((28, 8))
    data_dl = np.zeros((28, 8))
    for lead_time_index in range(28):
        shp = "/cw3e/mead/projects/cwp190/projects/post-processing/data/CNRFC/basin_information/CNRFCbasins/CNRFC_Basins_Sub.shp"
        gdf = gpd.read_file(shp)
        gdf_unique = gdf.drop_duplicates(subset=["ForecastGr"], keep="first")["ForecastGr"].dropna().tolist()
        basin_group_name = gdf_unique[basin_group]
        wrf_path = '/cw3e/mead/projects/cwp190/projects/post-processing/output/v9.7/westWRF/basin/lead_time_' + str(lead_time[lead_time_index]) + '/westWRF_ensemble_metrics_lead' + str(lead_time[lead_time_index]) + '_by_each_basin_range_-2.5_to_2.5.csv'
        dl_path = '/cw3e/mead/projects/cwp190/projects/post-processing/output/v9.7/DL_out/' + v2 + exp + '/basin/lead_' + str(lead_time[lead_time_index]) + '/DL_quantile_metrics_lead' + str(lead_time[lead_time_index]) + '_by_each_basin_range_-2.5_to_2.5.csv'
        wrf = pd.read_csv(wrf_path)
        dl = pd.read_csv(dl_path)
        data_wrf[lead_time_index, 0] = lead_time[lead_time_index]
        data_wrf[lead_time_index, 1] = wrf["mean_PIT"][basin_group]
        data_wrf[lead_time_index, 2] = wrf["REL"][basin_group]
        data_wrf[lead_time_index, 3] = wrf["RES"][basin_group]
        data_wrf[lead_time_index, 4] = wrf["BS"][basin_group]
        data_wrf[lead_time_index, 5] = wrf["BSS"][basin_group]
        data_wrf[lead_time_index, 6] = wrf["sharpness"][basin_group]
        data_wrf[lead_time_index, 7] = wrf["CRPSS"][basin_group]

        data_dl[lead_time_index, 0] = lead_time[lead_time_index]
        data_dl[lead_time_index, 1] = dl["mean_PIT"][basin_group]
        data_dl[lead_time_index, 2] = dl["REL"][basin_group]
        data_dl[lead_time_index, 3] = dl["RES"][basin_group]
        data_dl[lead_time_index, 4] = dl["BS"][basin_group]
        data_dl[lead_time_index, 5] = dl["BSS"][basin_group]
        data_dl[lead_time_index, 6] = dl["sharpness"][basin_group]
        data_dl[lead_time_index, 7] = dl["CRPSS"][basin_group]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    for idx in range(7):
        ax = axes[idx]
        ax.plot(data_wrf[1:, 0], data_wrf[1:, idx + 1], marker="o", label="WestWRF", linewidth=2)
        ax.plot(data_dl[1:, 0], data_dl[1:, idx + 1], marker="s", label="DL-QRNN", linewidth=2)

        ax.set_title(metrics[idx] + ' basin:' + basin_group_name, fontsize=14)
        ax.set_xlabel("Lead Time (hours)", fontsize=12)
        ax.set_ylabel(metrics[idx], fontsize=12)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"basin_{basin_group_name}_{v2}{exp}all_lead_time_range_-2.5_to_2.5.png"), dpi=150)
