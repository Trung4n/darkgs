#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, linear_to_srgb
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.shading import ShadingModel
import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def train_tensors(gaussians, shader):
    return {"xyz": gaussians._xyz, "albedo": gaussians._albedo, "normal": gaussians._normal, "opacity": gaussians._opacity,
            "scaling": gaussians._scaling, "rotation": gaussians._rotation,
            "shader.ambient_light_log": shader.ambient_light_log, "shader.scaling_factor": shader.scaling_factor}

def step_is_finite(loss, gaussians, shader):
    """True if the loss, every trainable tensor and every gradient is finite (a single GPU sync)."""
    ts = [loss.detach()]
    for p in train_tensors(gaussians, shader).values():
        ts.append(p.detach())
        if p.grad is not None:
            ts.append(p.grad)
    return bool(torch.stack([torch.isfinite(t).all() for t in ts]).all())

def nonfinite_report(iteration, cam_name, loss, image, gaussians, shader):
    lines = ["Non-finite value at iteration {} (camera {}): loss = {}, non-finite pixels in the render = {}".format(
        iteration, cam_name, loss.item(), int((~torch.isfinite(image)).sum()))]
    for name, p in train_tensors(gaussians, shader).items():
        for what, t in (("value", p.detach()), ("grad", p.grad)):
            if t is None:
                continue
            fin = t[torch.isfinite(t)]
            rng = "min {:.3e} max {:.3e}".format(fin.min().item(), fin.max().item()) if fin.numel() else "no finite entry"
            lines.append("  {:26s} {:5s} non-finite {:>8d} / {:<8d} finite range: {}".format(name, what, int(t.numel() - fin.numel()), t.numel(), rng))
    lines.append("Only grads non-finite: this iteration's backward pass produced the NaN/inf (parameters are still finite). Values non-finite: an earlier step poisoned them. "
                 "Re-run with --detect_anomaly to get the PyTorch op that produced it (or --no_check_finite to disable this check).")
    return "\n".join(lines)

def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, debug_from, wandb_cfg=None, light_params="model_parameters.pth", scaling_factor=0.1, check_finite=True):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, wandb_cfg)
    wandb_log_interval = wandb_cfg["log_interval"] if wandb_cfg else 10
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    shader = ShadingModel(light = "1DMLP")
    shader = shader.cuda()

    dict = torch.load(light_params)
    res = shader.load_state_dict(dict['model_state_dict'])

    shader = shader.to("cuda:0")
    r_vec = dict['so3'].squeeze()
    t_vec = dict['model_state_dict']['light._t_vec'].squeeze()
    print(r_vec)
    print(t_vec)

    # An initial (human) guess of scaling factor (by looking at the colmap vizualization).
    # SfM poses are only defined up to scale; for poses that are already metric (e.g. simulated data) use --scaling_factor 1.0.
    shader.set_scaling_factor(scaling_factor)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    warmup_factors  = torch.linspace(0, 1, opt.warmup_until_itr-opt.warmup_start_itr, device="cuda:0", requires_grad=False)
    shader.warmup_factor = 0.0
    for iteration in range(first_iter, opt.iterations + 1):     

        # WARM-UP STAGE ########################################
        if iteration >= opt.warmup_start_itr and iteration < opt.warmup_until_itr:
            warmup_factor = warmup_factors[iteration-opt.warmup_start_itr]
            # shader.warmup_factor = warmup_factor
            shader.light.set_r_vec(tuple([r_vec[0]*warmup_factor, r_vec[1]*warmup_factor, r_vec[2]*warmup_factor]))
            shader.light.set_t_vec(tuple([t_vec[0]*warmup_factor, t_vec[1]*warmup_factor, t_vec[2]*warmup_factor]))      
        ########################################################

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, shader, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            print("Gaussian Size: ", scene.gaussians.get_size)
            print("Scaling factor: ", shader.scaling_factor.item())
            print("Ambient Light: ", shader.ambient_light.item())
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, shader=shader)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = Ll1  
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        # Stop at the FIRST non-finite loss / parameter / gradient instead of silently training 30k iterations of NaN
        if check_finite and not step_is_finite(loss, gaussians, shader):
            raise RuntimeError(nonfinite_report(iteration, viewpoint_cam.image_name, loss, image, gaussians, shader))

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            iter_time = iter_start.elapsed_time(iter_end)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render, (pipe, background, shader), dataset.linearize)
            if iteration % wandb_log_interval == 0:
                wandb.log({
                    "iteration": iteration,
                    "train/l1_loss": Ll1.item(),
                    "train/total_loss": loss.item(),
                    "train/ema_loss": ema_loss_for_log,
                    "train/iter_time": iter_time,
                    "train/total_points": scene.gaussians.get_xyz.shape[0],
                    "train/scaling_factor": shader.scaling_factor.item(),
                    "train/ambient_light": shader.ambient_light.item(),
                })
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Some magic hyperparameters here
                    size_threshold = None if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent*shader.scaling_factor, size_threshold)
                    
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if iteration < opt.shader_optimize_until:
                shader.optimizer.step()
                shader.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                save_dict = {
                    'model_state_dict': shader.state_dict(),
                    'so3': shader.light._r_l2c_SO3.log()}
                res = torch.save(save_dict, scene.model_path + "/shader" + str(iteration) + ".pth")
                print("Parameters saved!")

def prepare_output_and_logger(args, wandb_cfg=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    # Create wandb run (mode="disabled" turns every wandb call into a no-op)
    wandb_cfg = wandb_cfg or {}
    wandb.init(
        project=wandb_cfg.get("project", "DarkGS"),
        name=wandb_cfg.get("name") or os.path.basename(os.path.normpath(args.model_path)),
        config=wandb_cfg.get("config", vars(args)),
        mode=None if wandb_cfg.get("use_wandb", False) else "disabled",
    )
    wandb.define_metric("iteration")
    wandb.define_metric("train/*", step_metric="iteration")
    wandb.define_metric("eval_test/*", step_metric="iteration")
    wandb.define_metric("eval_train/*", step_metric="iteration")
    wandb.define_metric("scene/*", step_metric="iteration")
    return tb_writer

def to_wandb_image(image, caption, srgb_display=False):
    # Keep raw intensities (wandb's own tensor path min-max normalizes, which distorts dark renders).
    # With --linearize the tensors are linear intensities: re-apply the sRGB curve so the logged images look like the inputs.
    if srgb_display:
        image = linear_to_srgb(image)
    return wandb.Image((image.clamp(0.0, 1.0) * 255).byte().permute(1, 2, 0).cpu().numpy(), caption=caption)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, srgb_display=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        wandb_log = {"iteration": iteration}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_srgb_test = 0.0
                wandb_renders, wandb_gts = [], []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if idx < 5:
                        wandb_renders.append(to_wandb_image(image, viewpoint.image_name, srgb_display))
                        if iteration == testing_iterations[0]:
                            wandb_gts.append(to_wandb_image(gt_image, viewpoint.image_name, srgb_display))
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if srgb_display:    # --linearize: PSNR above is on linear intensities; also report it in sRGB, comparable with methods evaluated on the input images
                        psnr_srgb_test += psnr(linear_to_srgb(image), linear_to_srgb(gt_image)).mean().double()
                psnr_test /= len(config['cameras'])
                psnr_srgb_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                wandb_log["eval_{}/l1_loss".format(config['name'])] = l1_test.item()
                wandb_log["eval_{}/psnr".format(config['name'])] = psnr_test.item()
                if srgb_display:
                    wandb_log["eval_{}/psnr_srgb".format(config['name'])] = psnr_srgb_test.item()
                wandb_log["eval_{}/render".format(config['name'])] = wandb_renders
                if wandb_gts:
                    wandb_log["eval_{}/ground_truth".format(config['name'])] = wandb_gts

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        opacity_values = scene.gaussians.get_opacity.detach().flatten()
        opacity_values = opacity_values[torch.isfinite(opacity_values)].cpu().numpy()      # wandb.Histogram raises on NaN/inf
        if opacity_values.size:
            wandb_log["scene/opacity_histogram"] = wandb.Histogram(opacity_values)
        wandb_log["scene/total_points"] = scene.gaussians.get_xyz.shape[0]
        wandb.log(wandb_log)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--light_params", type=str, default="model_parameters.pth", help="light/shading parameters (state dict + so3) from light calibration")
    parser.add_argument("--scaling_factor", type=float, default=0.1, help="initial scene scale of the shader; 1.0 for poses that are already metric")
    parser.add_argument("--no_check_finite", action="store_true", default=False, help="do not stop at the first non-finite loss / parameter / gradient")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="DarkGS")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_log_interval", type=int, default=10)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    wandb_cfg = {
        "use_wandb": args.use_wandb,
        "project": args.wandb_project,
        "name": args.wandb_name,
        "log_interval": max(1, args.wandb_log_interval),
        "config": vars(args),
    }
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb_cfg, args.light_params, args.scaling_factor, not args.no_check_finite)
    wandb.finish()

    # All done
    print("\nTraining complete.")

    def load_model_param(self)->None:
        print("Loading model parameters...")
        dict = torch.load('model_parameters.pth')
        res = self.shading_model.load_state_dict(dict['model_state_dict'])
        print("loaded model parameters: \n", self.shading_model.state_dict())
        print("load res: \n", res)
        r_vec = dict['so3'].squeeze()
        print(r_vec)
        self.shading_model.light.set_r_vec(tuple([r_vec[0], r_vec[1], r_vec[2]]))
        if hasattr(self.shading_model.light, 'sigma'):
            if self.shading_model.light.sigma.ndim == 0:
                self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma, 0])
            else:
                self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma[0], self.shading_model.light.sigma[1]])
        else:
            self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [0, 0])

