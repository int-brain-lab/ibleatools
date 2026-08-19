from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import torch
import torch.nn.functional as F

from ibl_style.style import figure_style
from ibl_style.utils import double_column_fig

# Loader swapped to this branch's publishing system; the figure body below is unchanged.
from _release import unit_release
from ephysatlas.unit_level_encoder.gmm_models import diag_log_prob, move


@dataclass
class FigureConfig:
    repo_id: str = "int-brain-lab/ea-encoder-unit"
    vintage: str = "2026_W26"
    token: Optional[str] = None
    cache_dir: Optional[Path] = None
    save_path: Path = Path("supp_figure2_unit_model_validation.pdf")
    dpi: int = 600
    seed: int = 0
    reconstruction_examples: int = 4
    kde_bandwidth_candidates: tuple[float, ...] = (0.15, 0.25, 0.40, 0.60, 0.80, 1.00)
    gaussian_floor_candidates: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    gaussian_shrinkage: float = 4.0


def _panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def _clean_axis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


@torch.no_grad()
def reconstruction_examples(model_ae, data, cfg, n, seed):
    rng=np.random.default_rng(seed); ids=np.flatnonzero(data.split==2)
    chosen=rng.choice(ids,min(n,len(ids)),replace=False)
    w=torch.from_numpy(data.waveforms[chosen]).to(cfg.device); a=torch.from_numpy(data.acgs[chosen]).to(cfg.device)
    rec=model_ae.reconstruct(model_ae.encode(w,a))
    return {"wave":data.waveforms[chosen],"wave_rec":rec["waveform_reconstruction"].cpu().numpy(),
            "acg":data.acgs[chosen],"acg_rec":rec["acg_reconstruction"].cpu().numpy()}


def draw_panel_a(fig, spec, examples):
    """
    Panel a: four held-out reconstruction examples.

    A dedicated header row is reserved for "Example 1", ..., "Example 4".
    This guarantees that the example labels sit ABOVE the plots rather than
    being positioned on top of the axes.

    Plot row 1: waveform Observation | Reconstruction
    Plot row 2: ACG Observation | Reconstruction
    """
    n = min(4, len(examples["wave"]))

    # Three rows:
    #   row 0: dedicated Example X headers
    #   row 1: waveform plots
    #   row 2: ACG plots
    #
    # Each example occupies two adjacent columns.
    gs = GridSpecFromSubplotSpec(
        3,
        2 * n,
        subplot_spec=spec,
        height_ratios=[0.16, 1.0, 1.0],
        wspace=0.10,
        hspace=0.30,
    )

    wf_lim = float(
        np.nanpercentile(
            np.abs(
                np.concatenate(
                    [
                        examples["wave"][:n].ravel(),
                        examples["wave_rec"][:n].ravel(),
                    ]
                )
            ),
            99,
        )
    )

    acg_lim = float(
        np.nanpercentile(
            np.concatenate(
                [
                    examples["acg"][:n].ravel(),
                    examples["acg_rec"][:n].ravel(),
                ]
            ),
            99,
        )
    )

    first = None

    for ex in range(n):
        # --------------------------------------------------------------
        # Dedicated header axis spanning the observation/reconstruction
        # pair. Because this is its own GridSpec row, "Example X" cannot
        # overlap the image axes below.
        # --------------------------------------------------------------
        ax_header = fig.add_subplot(gs[0, 2 * ex:2 * ex + 2])
        ax_header.axis("off")
        ax_header.text(
            0.5,
            0.55,
            f"Example {ex + 1}",
            transform=ax_header.transAxes,
            ha="center",
            va="center",
            fontweight="bold",
        )

        # --------------------------------------------------------------
        # Waveform observation + reconstruction
        # --------------------------------------------------------------
        ax_w = fig.add_subplot(gs[1, 2 * ex])
        ax_wr = fig.add_subplot(gs[1, 2 * ex + 1])

        if first is None:
            first = ax_w

        ax_w.imshow(
            examples["wave"][ex],
            aspect="auto",
            interpolation="nearest",
            cmap="seismic",
            vmin=-wf_lim,
            vmax=wf_lim,
        )
        ax_wr.imshow(
            examples["wave_rec"][ex],
            aspect="auto",
            interpolation="nearest",
            cmap="seismic",
            vmin=-wf_lim,
            vmax=wf_lim,
        )

        _clean_axis(ax_w)
        _clean_axis(ax_wr)

        ax_w.set_title("Observation", pad=3)
        ax_wr.set_title("Reconstruction", pad=3)

        # --------------------------------------------------------------
        # ACG observation + reconstruction
        # --------------------------------------------------------------
        ax_a = fig.add_subplot(gs[2, 2 * ex])
        ax_ar = fig.add_subplot(gs[2, 2 * ex + 1])

        ax_a.imshow(
            examples["acg"][ex],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=acg_lim,
        )
        ax_ar.imshow(
            examples["acg_rec"][ex],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=acg_lim,
        )

        _clean_axis(ax_a)
        _clean_axis(ax_ar)

        ax_a.set_title("Observation", pad=3)
        ax_ar.set_title("Reconstruction", pad=3)

    _panel_label(first, "a")


def _model_log_prob_from_logits(model,batch,logits):
    b,t,d=batch["target_z"].shape
    flat=batch["target_z"].reshape(-1,d)
    comp=diag_log_prob(flat,model.means,model.log_var).reshape(b,t,-1)
    lp=torch.logsumexp(F.log_softmax(logits,-1)[:,None]+comp,-1)
    return lp


@torch.no_grad()
def nll_by_neighbor_count(model, loader, cfg):
    """
    Held-out NLL as a function of discrete neighbor-count bins:

        0
        1-8
        9-16
        17-24
        25-32
        33-40
        41-48
        49-56
        57-64
    """
    bins = [
        0,
        (1, 8),
        (9, 16),
        (17, 24),
        (25, 32),
        (33, 40),
        (41, 48),
        (49, 56),
        (57, 64),
    ]

    bucket_sum = {key: [0.0, 0] for key in bins}

    model.eval()

    for raw in loader:
        batch = move(raw, cfg.device)

        logits = model.logits(
            batch["neighbor_z"],
            batch["relative_position"],
            batch["context"],
            batch["neighbor_padding_mask"],
        )
        lp = _model_log_prob_from_logits(
            model,
            batch,
            logits,
        )

        counts = batch["neighbor_count"].cpu().numpy()

        for i, count in enumerate(counts):
            valid = batch["target_mask"][i]
            vals = lp[i][valid]

            count = int(count)

            if count <= 0:
                key = 0
            else:
                # The model is configured for at most 64 neighbors. Clamp
                # defensively so any unexpected larger value still lands in
                # the final displayed bin.
                count = min(count, 64)
                lo = 1 + 8 * ((count - 1) // 8)
                hi = min(lo + 7, 64)
                key = (lo, hi)

            bucket_sum[key][0] += float((-vals).sum().cpu())
            bucket_sum[key][1] += int(len(vals))

    labels = []
    values = []
    ns = []

    for key in bins:
        total, n = bucket_sum[key]

        if key == 0:
            labels.append("0")
        else:
            labels.append(f"{key[0]}-{key[1]}")

        values.append(total / max(n, 1))
        ns.append(n)

    return (
        labels,
        np.asarray(values, dtype=float),
        np.asarray(ns, dtype=int),
    )


def _fallback_global_lp(model,target):
    comp=diag_log_prob(target,model.means,model.log_var)
    return torch.logsumexp(F.log_softmax(model.prior_logits,-1)[None]+comp,-1)


def _local_gaussian_lp(batch,floor,shrinkage,model):
    outputs=[]
    for i in range(len(batch["target_z"])):
        target=batch["target_z"][i][batch["target_mask"][i]]
        valid=~batch["neighbor_padding_mask"][i]
        nz=batch["neighbor_z"][i][valid]
        if len(nz)==0:
            outputs.append(_fallback_global_lp(model,target)); continue
        pos=batch["relative_position"][i][valid]
        w=torch.exp(-2.0*torch.linalg.norm(pos,dim=-1)); w=w/w.sum().clamp_min(1e-8)
        mean=(nz*w[:,None]).sum(0)
        var=((nz-mean).square()*w[:,None]).sum(0)
        eff=1.0/(w.square().sum().clamp_min(1e-8))
        alpha=eff/(eff+float(shrinkage))
        var=alpha*var+(1-alpha)*torch.ones_like(var)
        var=var.clamp_min(float(floor)**2)
        lp=-0.5*(np.log(2*np.pi)+torch.log(var)[None]+(target-mean[None]).square()/var[None]).sum(-1)
        outputs.append(lp)
    return torch.cat(outputs)


def _neighbor_kde_lp(batch,bandwidth,model):
    outputs=[]; h2=float(bandwidth)**2
    for i in range(len(batch["target_z"])):
        target=batch["target_z"][i][batch["target_mask"][i]]
        valid=~batch["neighbor_padding_mask"][i]; nz=batch["neighbor_z"][i][valid]
        if len(nz)==0:
            outputs.append(_fallback_global_lp(model,target)); continue
        pos=batch["relative_position"][i][valid]
        logw=-2.0*torch.linalg.norm(pos,dim=-1); logw=logw-torch.logsumexp(logw,dim=0)
        diff=(target[:,None,:]-nz[None,:,:]).square().sum(-1)
        kernel=-0.5*(target.shape[1]*np.log(2*np.pi*h2)+diff/h2)
        outputs.append(torch.logsumexp(logw[None]+kernel,dim=1))
    return torch.cat(outputs)


@torch.no_grad()
def _mean_nll_baseline(model,loader,cfg,kind,param,shrinkage=4.0):
    total=0.0;n=0
    for raw in loader:
        batch=move(raw,cfg.device)
        lp=_local_gaussian_lp(batch,param,shrinkage,model) if kind=="gaussian" else _neighbor_kde_lp(batch,param,model)
        total+=float((-lp).sum().cpu());n+=len(lp)
    return total/max(n,1)


def tune_density_baselines(model,val_loader,cfg,fig_cfg):
    g_scores={v:_mean_nll_baseline(model,val_loader,cfg,"gaussian",v,fig_cfg.gaussian_shrinkage) for v in fig_cfg.gaussian_floor_candidates}
    k_scores={v:_mean_nll_baseline(model,val_loader,cfg,"kde",v) for v in fig_cfg.kde_bandwidth_candidates}
    return min(g_scores,key=g_scores.get),min(k_scores,key=k_scores.get)


@torch.no_grad()
def ablation_log_likelihood(
    model,
    loader,
    cfg,
    gaussian_floor,
    kde_bandwidth,
    fig_cfg,
):
    """
    Mean held-out log likelihood per unit for the model variants/baselines.

    This is exactly the negative of the former NLL metric, so higher values are
    better.  We keep log likelihood rather than exponentiating to raw
    likelihood because the latter can underflow and is difficult to compare in
    high-dimensional latent space.
    """
    rng = np.random.default_rng(fig_cfg.seed + 991)

    totals = {
        k: [0.0, 0]
        for k in (
            "Full model",
            "Context only",
            "Neighbors only",
            "Global prior",
            "Shuffled context",
            "Shuffled neighbors",
        )
    }

    model.eval()

    for raw in loader:
        batch = move(raw, cfg.device)
        b = len(batch["context"])

        full = model.logits(
            batch["neighbor_z"],
            batch["relative_position"],
            batch["context"],
            batch["neighbor_padding_mask"],
        )

        allpad = torch.ones_like(
            batch["neighbor_padding_mask"]
        )

        variants = {
            "Full model": full,
            "Context only": model.logits(
                torch.zeros_like(batch["neighbor_z"]),
                torch.zeros_like(batch["relative_position"]),
                batch["context"],
                allpad,
            ),
            "Neighbors only": model.logits(
                batch["neighbor_z"],
                batch["relative_position"],
                torch.zeros_like(batch["context"]),
                batch["neighbor_padding_mask"],
            ),
            "Global prior": model.prior_logits[None].expand(b, -1),
        }

        perm = torch.as_tensor(
            rng.permutation(b),
            device=cfg.device,
            dtype=torch.long,
        )

        variants["Shuffled context"] = model.logits(
            batch["neighbor_z"],
            batch["relative_position"],
            batch["context"][perm],
            batch["neighbor_padding_mask"],
        )

        variants["Shuffled neighbors"] = model.logits(
            batch["neighbor_z"][perm],
            batch["relative_position"][perm],
            batch["context"],
            batch["neighbor_padding_mask"][perm],
        )

        for name, logits in variants.items():
            lp = _model_log_prob_from_logits(
                model,
                batch,
                logits,
            )[batch["target_mask"]]

            totals[name][0] += float(lp.sum().cpu())
            totals[name][1] += len(lp)

    scores = {
        name: total / max(n, 1)
        for name, (total, n) in totals.items()
    }

    # _mean_nll_baseline returns NLL; negate to get mean log likelihood.
    local_gaussian_ll = -_mean_nll_baseline(
        model,
        loader,
        cfg,
        "gaussian",
        gaussian_floor,
        fig_cfg.gaussian_shrinkage,
    )
    kde_ll = -_mean_nll_baseline(
        model,
        loader,
        cfg,
        "kde",
        kde_bandwidth,
    )

    # Requested ordering: swap the previous KDE / Local Gaussian positions.
    # Put KDE before Local Gaussian.
    scores["KDE"] = kde_ll
    scores["Local Gaussian"] = local_gaussian_ll

    return scores


def draw_panel_b(ax,labels,values,ns):
    x=np.arange(len(labels)); ax.plot(x,values,marker="o",ms=3,lw=1)
    ax.set_xticks(x,labels,rotation=55,ha="right"); ax.set_ylabel("Held-out NLL / unit"); ax.set_xlabel("Neighbor count")
    ax.spines[["top","right"]].set_visible(False); _panel_label(ax,"b")


def draw_panel_c(ax, scores):
    names = list(scores)
    values = np.asarray(
        [scores[n] for n in names],
        dtype=float,
    )
    x = np.arange(len(names))

    ax.bar(
        x,
        values,
    )
    ax.set_xticks(
        x,
        names,
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Held-out log likelihood / unit")

    # Since the differences are relatively small, use a tight but padded range
    # rather than forcing the bars to start at zero.
    finite = values[np.isfinite(values)]
    if finite.size:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        span = max(hi - lo, 1e-6)
        ax.set_ylim(
            lo - 0.12 * span,
            hi + 0.12 * span,
        )

    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "c")


def make_supp_figure2(fig_cfg=FigureConfig()):
    figure_style()
    cfg,data,model_ae,model_gmm,scaler,standardized,datasets,loaders=unit_release(fig_cfg.repo_id,fig_cfg.vintage,cache_dir=fig_cfg.cache_dir)
    examples=reconstruction_examples(model_ae,data,cfg,fig_cfg.reconstruction_examples,fig_cfg.seed)
    labels,count_nll,ns=nll_by_neighbor_count(model_gmm,loaders[2],cfg)
    gaussian_floor,kde_bw=tune_density_baselines(model_gmm,loaders[1],cfg,fig_cfg)
    scores=ablation_log_likelihood(model_gmm,loaders[2],cfg,gaussian_floor,kde_bw,fig_cfg)
    print(f"validation-selected local Gaussian sigma floor={gaussian_floor}")
    print(f"validation-selected KDE bandwidth={kde_bw}")
    print(scores)

    fig=double_column_fig(); fig.set_size_inches(fig.get_size_inches()[0]*1.08,8.8)
    outer=fig.add_gridspec(3,1,height_ratios=[2.9,1.25,1.45],hspace=0.50)
    draw_panel_a(fig,outer[0],examples)
    axb=fig.add_subplot(outer[1]); draw_panel_b(axb,labels,count_nll,ns)
    axc=fig.add_subplot(outer[2]); draw_panel_c(axc,scores)
    fig.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)
    fig_cfg.save_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(fig_cfg.save_path,dpi=fig_cfg.dpi,bbox_inches="tight",pad_inches=0.02);plt.close(fig)
    np.savez_compressed(fig_cfg.save_path.with_suffix(".npz"),neighbor_labels=np.asarray(labels),neighbor_nll=count_nll,
                        neighbor_target_units=ns,ablation_names=np.asarray(list(scores)),ablation_log_likelihood=np.asarray(list(scores.values())),
                        gaussian_floor=gaussian_floor,kde_bandwidth=kde_bw)
    print(f"saved: {fig_cfg.save_path}")


if __name__=="__main__":
    make_supp_figure2()