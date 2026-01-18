from functools import partial
from typing import Callable, List, Optional, Union, Tuple

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
# from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler

# 继承自 ModifiedStableDiffusionPipeline，这让我们能利用那里定义的辅助函数
from modified_stable_diffusion import ModifiedStableDiffusionPipeline
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt



### credit to: https://github.com/cccntu/efficient-prompt-to-prompt
# 这是一个辅助函数，实现了 DDIM 的一步更新公式
# 它可以用于去噪（Noise -> Image）也可以用于加噪（Image -> Noise），取决于传入的 alpha 参数顺序
def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """ from noise to image"""
    # DDIM 的确定性迭代公式
    # x_{t-1} = sqrt(alpha_{t-1}) * ( (x_t - sqrt(1-alpha_t)*eps) / sqrt(alpha_t) ) + sqrt(1-alpha_{t-1}) * eps
    # 代码里对公式做了一些数学变形以提高计算稳定性
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

# 前向 DDIM 实际上就是调用 backward_ddim，但是会在调用逻辑里把 alpha 参数反过来传
def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)

# [核心类] 支持反演的 Stable Diffusion 管道
class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = False,
        image_encoder=None
    ):

        # 初始化父类
        super(InversableStableDiffusionPipeline, self).__init__(vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                image_encoder=image_encoder)

        # [关键设计] 定义 forward_diffusion (反演/加噪过程)
        # 它是 backward_diffusion 的一个偏函数 (partial)，只是强制把 reverse_process 设为 True
        # 也就是说，这个类用同一个函数 backward_diffusion 处理生成和反演，只通过参数控制方向
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
        self.count = 0
    
    # 获取随机初始噪声 z_T
    def get_random_latents(self, latents=None, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1
        device = self._execution_device

        num_channels_latents = self.unet.in_channels

        # 调用父类的 prepare_latents 生成标准高斯噪声
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    # 辅助函数：获取 Prompt 的文本嵌入
    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        # Tokenizer 编码
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        # Text Encoder 提取特征
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    
    # 辅助函数：将图片编码为 Latents (z_0)
    # 这在反演开始前必须调用，把像素图片转为潜空间表示
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode() # 通常用 mode 获取确定性结果
        latents = encoding * 0.18215 # 缩放因子
        return latents


    # [核心函数] 统一的扩散处理函数 (既能生成，也能反演)
    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: True = False, # [开关] True=反演(Inversion), False=生成(Generation)
        **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # 判断是否使用 Classifier-Free Guidance (CFG)
        # 注意：在 Gaussian Shading 的反演阶段，通常 guidance_scale=1 (不使用 CFG)
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        # 移动 timesteps 到设备 (GPU)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        # 缩放初始噪声 (对于 DDIM 通常是 1)
        latents = latents * self.scheduler.init_noise_sigma

        # Prompt-to-Prompt 的相关逻辑 (用于编辑任务，水印实验中主要关注 else 分支)
        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False


        # [时间步循环]
        # 如果是反演 (reverse_process=True)，我们需要从 t=0 走到 t=T，所以要 reversed(timesteps)
        # 如果是生成 (reverse_process=False)，我们需要从 t=T 走到 t=0，正常遍历 timesteps
        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # 扩展 latents 用于 CFG 预测 (Uncond + Cond)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 1. 使用 UNet 预测噪声残差 noise_pred (epsilon)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # 2. 执行 CFG 引导
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # 计算上一个时间步 (仅用于获取 alpha，逻辑稍显复杂是为了兼容不同 Scheduler)
            prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )
            # call the callback, if provided # 回调函数
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
            # ddim # [核心 DDIM 更新逻辑]
            # 获取当前步的 alpha_cumprod
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            # 获取前一步的 alpha_cumprod (处理边界情况)
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            # [关键] 如果是反演过程 (reverse_process=True)
            # 我们需要交换 alpha_t 和 alpha_t_prev 的角色
            # 因为我们是往“加噪”的方向走，当前的 alpha 实际上是“较小”的那个，目标是“较大”的那个
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
                
            # 调用数学公式更新 Latents
            # 无论方向如何，都使用同一个公式，只是输入的 alpha 参数不同
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents

    # 后面是两个辅助函数：解码图片和 Tensor 转 Numpy，与父类类似
    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image
