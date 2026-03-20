"""Training loop for HelmholtzPINN: Adam + L-BFGS with wandb logging."""

import torch
import os
from tqdm import tqdm
from losses import total_loss
from evaluate import evaluate_against_analytic

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(config):
    if not config.use_wandb or wandb is None:
        return None
    run = wandb.init(
        project=config.wandb_project,
        config=config.to_dict(),
        name=f"ka={config.ka:.2f}",
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
    """Sample all point sets for training."""
    interior = domain.sample_interior(config.n_interior)
    boundary = domain.sample_scatterer_boundary(config.n_boundary)
    outer = domain.sample_outer_boundary(config.n_outer)
    return interior, boundary, outer


def save_checkpoint(model, config, phase, output_dir="checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"helmholtz_ka{config.ka:.2f}_{phase}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Checkpoint saved: {path}")
    return path


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

    pbar = tqdm(range(1, config.adam_epochs + 1), desc="Adam", ncols=100)
    for epoch in pbar:
        # Resample interior points periodically
        if epoch > 1 and epoch % config.resample_every == 0:
            interior = domain.sample_interior(config.n_interior)

        optimizer.zero_grad()
        loss, loss_pde, loss_bc, loss_abc = total_loss(
            model, interior, boundary, outer, config.k, config
        )
        loss.backward()
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

    save_checkpoint(model, config, "lbfgs")
    return model


def train(model, domain, config, analytic_fn=None):
    """Full training pipeline: Adam followed by L-BFGS."""
    init_wandb(config)

    model = train_adam(model, domain, config, analytic_fn)
    model = train_lbfgs(model, domain, config, analytic_fn,
                        start_epoch=config.adam_epochs)

    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()

    return model
