"""Training loop for HelmholtzPINN: Adam + L-BFGS with wandb logging."""

import torch
import os
from tqdm import tqdm
from .losses import total_loss
from .evaluate import evaluate_against_analytic

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(config, run_name=None):
    if not config.use_wandb or wandb is None:
        return None
    run = wandb.init(
        project=config.wandb_project,
        config=config.to_dict(),
        name=run_name or _checkpoint_tag(config),
    )
    return run


def _log(metrics, step, config):
    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def _evaluate_and_log(model, domain, config, analytic_fn, step, phase):
    metrics = evaluate_against_analytic(model, domain, config, analytic_fn)
    metrics["phase"] = 0 if phase == "adam" else 1
    _log({f"eval/{k}": v for k, v in metrics.items()}, step, config)
    return metrics


def _sample_points(domain, config):
    """Sample all point sets for training, dispatching based on config."""
    interior = domain.sample_interior(
        config.n_interior,
        strategy=config.sampling_strategy,
        alpha=config.radial_alpha,
        outer_boundary=config.outer_boundary,
        cluster_radius=getattr(config, 'cluster_radius', None),
    )
    boundary = domain.sample_scatterer_boundary(config.n_boundary)
    if config.outer_boundary == "circle":
        outer = domain.sample_circular_outer_boundary(config.n_outer)
    else:
        outer = domain.sample_outer_boundary(config.n_outer)
    return interior, boundary, outer


def _checkpoint_tag(config):
    """Build a descriptive tag for checkpoint filenames and artifact names."""
    from datetime import datetime
    ts = datetime.now().strftime("%m%d_%H%M")
    tag = f"ka{config.ka:.2f}_L{config.L:.1f}"
    if getattr(config, "use_honeycomb", False):
        tag = f"hc_{tag}"
    if getattr(config, "outer_boundary", "square") == "circle":
        tag += "_circ"
    if getattr(config, "abc_order", 1) == 2:
        tag += "_bgt2"
    tag += f"_{ts}"
    return tag


def save_checkpoint(model, config, phase, output_dir="checkpoints", log_to_wandb=False):
    os.makedirs(output_dir, exist_ok=True)
    tag = _checkpoint_tag(config)
    path = os.path.join(output_dir, f"helmholtz_{tag}_{phase}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Checkpoint saved: {path}")

    if log_to_wandb and wandb is not None and wandb.run is not None:
        artifact = wandb.Artifact(
            f"helmholtz-{tag}-{phase}",
            type="model",
            metadata={"ka": config.ka, "L": config.L, "phase": phase,
                       "honeycomb": getattr(config, "use_honeycomb", False)},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        print(f"  Artifact logged to wandb: {artifact.name}")

    return path


def _blend_interior(old, new):
    """Blend 50% old + 50% new interior points for smoother resampling."""
    n = old["x"].shape[0]
    half = n // 2
    perm = torch.randperm(n, device=old["x"].device)
    keep = perm[:half]
    x = torch.cat([old["x"][keep].detach(), new["x"][:n - half]])
    y = torch.cat([old["y"][keep].detach(), new["y"][:n - half]])
    return {"x": x.requires_grad_(True), "y": y.requires_grad_(True)}


def _compute_pointwise_pde_residual(model, points, k):
    """Compute per-point PDE residual magnitude (no graph needed — for RAD sampling only)."""
    x = points["x"].detach().requires_grad_(True)
    y = points["y"].detach().requires_grad_(True)

    u, v = model(x, y)
    ones = torch.ones_like(u)

    du_dx = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, ones, create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, ones, create_graph=True, retain_graph=True)[0]

    d2u_dx2 = torch.autograd.grad(du_dx, x, ones, create_graph=False, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, ones, create_graph=False, retain_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx, x, ones, create_graph=False, retain_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy, y, ones, create_graph=False, retain_graph=False)[0]

    res_u = d2u_dx2 + d2u_dy2 + k ** 2 * u.detach()
    res_v = d2v_dx2 + d2v_dy2 + k ** 2 * v.detach()

    return torch.sqrt(res_u ** 2 + res_v ** 2).detach()


def train_adam(model, domain, config, analytic_fn=None):
    """Train with Adam optimizer + cosine annealing scheduler."""
    print(f"=== Adam phase: {config.adam_epochs} epochs ===")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.adam_epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    interior, boundary, outer = _sample_points(domain, config)
    grad_clip = getattr(config, "grad_clip", 1.0)

    pbar = tqdm(range(1, config.adam_epochs + 1), desc="Adam", ncols=100)
    for epoch in pbar:
        # Resample interior points periodically (blended for stability)
        if epoch > 1 and epoch % config.resample_every == 0:
            if config.use_rad:
                # RAD: generate 3x candidates, sample proportional to residual
                candidates = domain.sample_interior(
                    config.n_interior * 3,
                    strategy=config.sampling_strategy,
                    alpha=config.radial_alpha,
                    outer_boundary=config.outer_boundary,
                    cluster_radius=getattr(config, 'cluster_radius', None),
                )
                residuals = _compute_pointwise_pde_residual(model, candidates, config.k)
                prob = residuals ** config.rad_k + config.rad_c
                prob = prob / prob.sum()
                indices = torch.multinomial(prob, config.n_interior, replacement=True)
                new_interior = {
                    "x": candidates["x"].detach()[indices].requires_grad_(True),
                    "y": candidates["y"].detach()[indices].requires_grad_(True),
                }
            else:
                new_interior = domain.sample_interior(
                    config.n_interior,
                    strategy=config.sampling_strategy,
                    alpha=config.radial_alpha,
                    outer_boundary=config.outer_boundary,
                    cluster_radius=getattr(config, 'cluster_radius', None),
                )
            interior = _blend_interior(interior, new_interior)

        optimizer.zero_grad()
        loss, loss_pde, loss_bc, loss_abc = total_loss(
            model, interior, boundary, outer, config.k, config
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Logging
        if epoch % 50 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            metrics = {
                "loss/total": loss.item(),
                "loss/pde": loss_pde.item(),
                "loss/bc": loss_bc.item(),
                "loss/abc": loss_abc.item(),
                "lr": lr,
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "phase": 0,
            }
            _log(metrics, epoch, config)
            pbar.set_postfix(
                loss=f"{loss.item():.4e}",
                pde=f"{loss_pde.item():.4e}",
                bc=f"{loss_bc.item():.4e}",
            )

        # Evaluation
        if analytic_fn and epoch % config.eval_every == 0:
            eval_metrics = _evaluate_and_log(model, domain, config, analytic_fn, epoch, "adam")
            tqdm.write(f"  [Adam {epoch}] L2_rel={eval_metrics['l2_rel']:.4e}, max_err={eval_metrics['max_err']:.4e}")

    save_checkpoint(model, config, "adam")
    return model


def train_lbfgs(model, domain, config, analytic_fn=None, start_epoch=0):
    """Train with L-BFGS optimizer (fixed collocation points)."""
    print(f"=== L-BFGS phase: {config.lbfgs_epochs} epochs ===")
    model.train()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=config.lbfgs_lr,
        max_iter=config.lbfgs_max_iter,
        history_size=config.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    # Fixed collocation points for L-BFGS (no resampling)
    interior, boundary, outer = _sample_points(domain, config)

    pbar = tqdm(range(1, config.lbfgs_epochs + 1), desc="L-BFGS", ncols=100)
    for epoch in pbar:
        global_epoch = start_epoch + epoch

        def closure():
            optimizer.zero_grad()
            loss, loss_pde, loss_bc, loss_abc = total_loss(
                model, interior, boundary, outer, config.k, config
            )
            loss.backward()
            # Store for logging
            closure.loss = loss.item()
            closure.loss_pde = loss_pde.item()
            closure.loss_bc = loss_bc.item()
            closure.loss_abc = loss_abc.item()
            return loss

        closure.loss = 0.0
        closure.loss_pde = 0.0
        closure.loss_bc = 0.0
        closure.loss_abc = 0.0

        optimizer.step(closure)

        metrics = {
            "loss/total": closure.loss,
            "loss/pde": closure.loss_pde,
            "loss/bc": closure.loss_bc,
            "loss/abc": closure.loss_abc,
            "phase": 1,
        }
        _log(metrics, global_epoch, config)
        pbar.set_postfix(
            loss=f"{closure.loss:.4e}",
            pde=f"{closure.loss_pde:.4e}",
            bc=f"{closure.loss_bc:.4e}",
        )

        if analytic_fn and epoch % 20 == 0:
            eval_metrics = _evaluate_and_log(model, domain, config, analytic_fn, global_epoch, "lbfgs")
            tqdm.write(f"  [LBFGS {epoch}] L2_rel={eval_metrics['l2_rel']:.4e}, max_err={eval_metrics['max_err']:.4e}")

    save_checkpoint(model, config, "lbfgs", log_to_wandb=config.use_wandb)
    return model


def train(model, domain, config, analytic_fn=None, run_name=None):
    """Full training pipeline: Adam followed by L-BFGS."""
    init_wandb(config, run_name=run_name)

    model = train_adam(model, domain, config, analytic_fn)
    model = train_lbfgs(model, domain, config, analytic_fn,
                        start_epoch=config.adam_epochs)

    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()

    return model
