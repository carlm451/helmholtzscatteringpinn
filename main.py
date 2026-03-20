"""CLI entry point for HelmholtzPINN."""

import argparse
import math
import torch

from helmholtz.config import HelmholtzConfig
from helmholtz.analytic import scattered_field, total_field
from helmholtz.domain import ScatteringDomain
from helmholtz.network import HelmholtzPINN
from helmholtz.train import train, train_adam, train_lbfgs, init_wandb, save_checkpoint
from helmholtz.evaluate import evaluate_against_analytic
from helmholtz.visualize import create_zoom_report, plot_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="HelmholtzPINN: 2D Helmholtz scattering solver")
    parser.add_argument("--ka", type=float, default=math.pi, help="ka value (default: pi)")
    parser.add_argument("--L", type=float, default=3.0, help="Domain half-size")
    parser.add_argument("--adam-epochs", type=int, default=None, help="Override Adam epochs")
    parser.add_argument("--lbfgs-epochs", type=int, default=None, help="Override L-BFGS epochs")
    parser.add_argument("--adam-lr", type=float, default=None, help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--no-lbfgs", action="store_true", help="Skip L-BFGS phase")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default=None, help="Device: mps, cuda, cpu")
    parser.add_argument("--eval-only", type=str, default=None, help="Path to checkpoint for eval only")
    parser.add_argument("--analytic-only", action="store_true", help="Only generate analytic solution plots")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    # Loss weights
    parser.add_argument("--lambda-bc", type=float, default=None, help="BC loss weight")
    parser.add_argument("--lambda-pde", type=float, default=None, help="PDE loss weight")
    parser.add_argument("--lambda-abc", type=float, default=None, help="ABC loss weight")
    # Network architecture
    parser.add_argument("--fourier-features", type=int, default=None, help="Fourier features count")
    parser.add_argument("--fourier-sigma", type=float, default=None, help="Fourier feature bandwidth")
    parser.add_argument("--activation", type=str, default=None, help="Activation: tanh or sin")
    parser.add_argument("--hidden-neurons", type=int, default=None, help="Neurons per layer")
    parser.add_argument("--hidden-layers", type=int, default=None, help="Hidden layers")
    # Sampling
    parser.add_argument("--n-interior", type=int, default=None, help="Interior collocation points")
    parser.add_argument("--n-boundary", type=int, default=None, help="Boundary points")
    parser.add_argument("--n-outer", type=int, default=None, help="Outer boundary points")
    parser.add_argument("--sampling", type=str, default=None, help="Sampling strategy: uniform, radial_bias, rad")
    parser.add_argument("--radial-alpha", type=float, default=None, help="Radial bias exponent")
    parser.add_argument("--use-rad", action="store_true", help="Enable RAD adaptive resampling")
    # Outer boundary / ABC
    parser.add_argument("--outer-boundary", type=str, default=None, help="Outer boundary shape: square or circle")
    parser.add_argument("--abc-order", type=int, default=None, help="ABC order: 1 or 2 (BGT2)")
    # wandb
    parser.add_argument("--run-name", type=str, default=None, help="wandb run name override")
    return parser.parse_args()


def get_config(args):
    """Build config from CLI args, using factory methods for known ka values."""
    ka = args.ka
    kwargs = {}
    if args.L != 3.0:
        kwargs["L"] = args.L
    if args.device:
        kwargs["device"] = args.device
    if args.no_wandb:
        kwargs["use_wandb"] = False

    # Use factory methods for standard ka values
    if abs(ka - math.pi) < 0.01:
        config = HelmholtzConfig.ka_pi(**kwargs)
    elif abs(ka - 2 * math.pi) < 0.01:
        config = HelmholtzConfig.ka_2pi(**kwargs)
    elif abs(ka - 3 * math.pi) < 0.01:
        config = HelmholtzConfig.ka_3pi(**kwargs)
    else:
        config = HelmholtzConfig(ka=ka, **kwargs)

    # CLI overrides
    if args.adam_lr is not None:
        config.adam_lr = args.adam_lr
    if args.adam_epochs is not None:
        config.adam_epochs = args.adam_epochs
    if args.lbfgs_epochs is not None:
        config.lbfgs_epochs = args.lbfgs_epochs

    # Extended overrides
    overrides = {
        "lambda_bc": args.lambda_bc,
        "lambda_pde": args.lambda_pde,
        "lambda_abc": args.lambda_abc,
        "n_fourier_features": args.fourier_features,
        "fourier_sigma": args.fourier_sigma,
        "activation": args.activation,
        "n_hidden_neurons": args.hidden_neurons,
        "n_hidden_layers": args.hidden_layers,
        "n_interior": args.n_interior,
        "n_boundary": args.n_boundary,
        "n_outer": args.n_outer,
        "sampling_strategy": args.sampling,
        "radial_alpha": args.radial_alpha,
        "outer_boundary": args.outer_boundary,
        "abc_order": args.abc_order,
    }
    if args.use_rad:
        overrides["use_rad"] = True
    for attr, val in overrides.items():
        if val is not None:
            setattr(config, attr, val)

    return config


def run_analytic_only(config, output_dir):
    """Generate analytic solution plots only."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os

    os.makedirs(output_dir, exist_ok=True)
    L = config.L
    grid_size = 300

    x = np.linspace(-L, L, grid_size)
    y = np.linspace(-L, L, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    phi = total_field(xx, yy, config.k, config.a, config.n_series_terms)

    for name, values in [("real", np.real(phi)), ("imag", np.imag(phi)), ("magnitude", np.abs(phi))]:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=values, x=x, y=y,
            colorscale="RdBu_r" if name != "magnitude" else "Viridis",
            zmid=0 if name != "magnitude" else None,
        ))
        theta = np.linspace(0, 2 * np.pi, 100)
        for cx, cy, r in config.scatterers:
            fig.add_trace(go.Scatter(
                x=cx + r * np.cos(theta), y=cy + r * np.sin(theta),
                mode="lines", line=dict(color="white", width=2), showlegend=False,
            ))
        fig.update_layout(
            title=f"Analytic Total Field ({name}) — ka={config.ka:.2f}",
            xaxis_title="x", yaxis_title="y",
            xaxis=dict(scaleanchor="y"), width=700, height=600,
        )
        path = os.path.join(output_dir, f"analytic_ka{config.ka:.2f}_{name}.html")
        fig.write_html(path)
        print(f"  Saved: {path}")


def main():
    args = parse_args()
    config = get_config(args)

    print(f"HelmholtzPINN — ka={config.ka:.4f}, k={config.k:.4f}, device={config.device}")
    print(f"  Network: {config.n_hidden_layers} layers x {config.n_hidden_neurons} neurons, "
          f"{config.n_fourier_features} Fourier features")
    print(f"  Sampling: {config.n_interior} interior, {config.n_boundary} BC, {config.n_outer} ABC")
    print(f"  Strategy: {config.sampling_strategy}, outer={config.outer_boundary}, "
          f"ABC order={config.abc_order}, RAD={config.use_rad}")

    if args.analytic_only:
        print("Generating analytic solution plots...")
        run_analytic_only(config, args.output_dir)
        return

    # Setup
    domain = ScatteringDomain(config.L, config.scatterers, config.device)
    model = HelmholtzPINN(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    analytic_fn = scattered_field

    if args.eval_only:
        print(f"Loading checkpoint: {args.eval_only}")
        model.load_state_dict(torch.load(args.eval_only, map_location=config.device, weights_only=True))
        metrics = evaluate_against_analytic(model, domain, config, analytic_fn)
        print(f"  L2 relative error: {metrics['l2_rel']:.6e}")
        print(f"  Max error: {metrics['max_err']:.6e}")
        print("Generating zoom report...")
        create_zoom_report(model, domain, config, analytic_fn, args.output_dir)
        return

    # Training
    run_name = args.run_name
    if args.no_lbfgs:
        init_wandb(config, run_name=run_name)
        model = train_adam(model, domain, config, analytic_fn)
        import wandb as wb
        if config.use_wandb and wb.run is not None:
            wb.finish()
    else:
        model = train(model, domain, config, analytic_fn, run_name=run_name)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    metrics = evaluate_against_analytic(model, domain, config, analytic_fn)
    print(f"  L2 relative error: {metrics['l2_rel']:.6e}")
    print(f"  L2 absolute error: {metrics['l2_abs']:.6e}")
    print(f"  Max error:         {metrics['max_err']:.6e}")
    print(f"  Mean error:        {metrics['mean_err']:.6e}")

    # Generate zoom report
    print("\nGenerating zoom report...")
    create_zoom_report(model, domain, config, analytic_fn, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
