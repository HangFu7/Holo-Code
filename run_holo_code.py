import argparse
import os
from tqdm import tqdm
import torch
import open_clip

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL

from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *


def main(args):
    # 1. 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 初始化调度器
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_path,
        subfolder='scheduler'
    )

    # 3. 初始化 Pipeline
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # 4. 加载 CLIP (可选)
    ref_model = None
    ref_clip_preprocess = None
    ref_tokenizer = None
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # 5. 加载数据集
    dataset, prompt_key = get_dataset(args)

    # 6. 算法选择
    if args.algo == 'holo':
        print(f"\n{'=' * 40}")
        print("🚀 Running Algorithm: Holo-Code")
        print(f"{'=' * 40}\n")
        watermark = Holo_Shading(
            args.channel_copy,
            args.hw_copy,
            args.fpr,
            args.user_number
        )

    elif args.algo == 'gs':
        print(f"\n{'=' * 40}")
        print("📉 Running Algorithm: Gaussian Shading (Baseline)")
        print(f"{'=' * 40}\n")
        if args.chacha:
            watermark = Gaussian_Shading_chacha(
                args.channel_copy,
                args.hw_copy,
                args.fpr,
                args.user_number
            )
        else:
            watermark = Gaussian_Shading(
                args.channel_copy,
                args.hw_copy,
                args.fpr,
                args.user_number
            )

    elif args.algo == 'gssync':
        print(f"\n{'=' * 40}")
        print("📈 Running Algorithm: GS + Blind Sync (Strong Baseline)")
        print(f"{'=' * 40}\n")
        watermark = Gaussian_Shading_Sync(
            args.channel_copy,
            args.hw_copy,
            args.fpr,
            args.user_number
        )

    elif args.algo == 'prc':
        print(f"\n{'=' * 40}")
        print("🧩 Running Algorithm: PRC Watermark (Baseline)")
        print(f"{'=' * 40}\n")
        watermark = PRC_Watermark(args)

    else:
        raise ValueError(f"Unknown algorithm type: {args.algo}")

    # 7. 创建目录
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "prompt"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "image"), exist_ok=True)

    # 8. 准备 Embeddings
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    acc = []
    clip_scores = []
    iot_decode_success = []
    iot_effective_loss = []
    iot_psnr = []

    # 加载 VAE 用于攻击（当前 image_utils.py 里的 vae_attack 使用的是 CompressAI，
    # 这里保留这个加载逻辑只是为了兼容你原代码，不参与实际 CompressAI 攻击）
    vae_model = None
    if args.vae_attack:
        print(">>> Loading VAE model for Attack...")
        try:
            vae_model = AutoencoderKL.from_pretrained(
                args.model_path,
                subfolder="vae"
            ).to(device)
        except Exception as e:
            print(f"Warning: failed to load VAE model: {e}")
            vae_model = None

    # ===================== 9. 开始循环 =====================
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        # --- A. 生成 (Watermarking) ---
        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()

        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]

        # 保存水印图
        image_w.save(os.path.join(args.output_path, "image", f"img_{i}_watermarked.png"))

        # --- B. 攻击 (Distortion) ---
        args._last_iot_stats = None
        image_w_distortion = image_distortion(image_w, seed, args, vae_model=vae_model)
        print("DEBUG _last_iot_stats =", getattr(args, "_last_iot_stats", None))
        
        if hasattr(args, '_last_iot_stats') and args._last_iot_stats is not None:
            iot_decode_success.append(args._last_iot_stats.decode_success)
            iot_effective_loss.append(args._last_iot_stats.effective_loss_rate)
            iot_psnr.append(args._last_iot_stats.psnr)

        # 判定攻击保存逻辑
        gs_attack_params = [
            args.jpeg_ratio,
            args.random_crop_ratio,
            args.random_drop_ratio,
            args.gaussian_blur_r,
            args.median_blur_k,
            args.resize_ratio,
            args.gaussian_std,
            args.sp_prob,
            args.brightness_factor,
            args.contrast_factor,
            args.translation_shift,
            args.perspective_scale,
            args.crop_scale_ratio,
            args.composite_crop_jpeg,
            1 if args.vae_attack else None,
            1 if args.iot_attack else None,
        ]

        is_attacked = any(x is not None for x in gs_attack_params)

        if is_attacked:
            attack_filename = os.path.join(args.output_path, "image", f"img_{i}_attacked.png")
            try:
                image_w_distortion.save(attack_filename)
            except Exception as e:
                print(f"Warning: Failed to save attacked image: {e}")

        # --- C. 提取 (Inversion) ---
        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        # --- D. 评估 (Evaluation) ---
        eval_result = watermark.eval_watermark(reversed_latents_w)

        if isinstance(eval_result, tuple):
            acc_metric = eval_result[0]
        else:
            acc_metric = eval_result

        acc.append(acc_metric)

        if args.reference_model is not None:
            score = measure_similarity(
                [image_w],
                current_prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                device
            )
            clip_score = score[0].item()
        else:
            clip_score = 0.0

        clip_scores.append(clip_score)

    # IoT 统计
    if len(iot_decode_success) > 0:
        mean_decode_success = sum(iot_decode_success) / len(iot_decode_success)
        mean_effective_loss = sum(iot_effective_loss) / len(iot_effective_loss)
        mean_psnr = sum(iot_psnr) / len(iot_psnr)

        print("\n" + "=" * 50)
        print("IoT Transmission Summary")
        print("=" * 50)
        print(f"Mean Decode Success Rate : {mean_decode_success:.4f}")
        print(f"Mean Effective Loss Rate : {mean_effective_loss:.4f}")
        print(f"Mean PSNR                : {mean_psnr:.4f}")
        print("=" * 50)

        with open(os.path.join(args.output_path, "iot_summary.txt"), "w") as f:
            f.write(f"mean_decode_success_rate: {mean_decode_success:.6f}\n")
            f.write(f"mean_effective_loss_rate: {mean_effective_loss:.6f}\n")
            f.write(f"mean_psnr: {mean_psnr:.6f}\n")



    # 10. 统计最终指标
    tpr_detection, tpr_traceability = watermark.get_tpr()
    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Holo-Code / GS / PRC Evaluation')

    # 基础参数
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')

    # 生成参数
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)

    # 水印参数
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--chacha', action='store_true', help='Use chacha20 for cipher in GS baseline')

    # 路径
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')

    # 攻击参数
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--translation_shift', type=int, default=None)
    parser.add_argument('--perspective_scale', type=float, default=None)

    parser.add_argument('--iot_channel_code', type=str, default='none',
                    choices=['none', 'ldpc'],
                    help='Channel coding scheme for IoT transmission')

    parser.add_argument('--iot_ldpc_k', type=int, default=2048,
                    help='LDPC information length per block')

    parser.add_argument('--iot_ldpc_m', type=int, default=1024,
                    help='LDPC parity length per block')

    parser.add_argument('--iot_ldpc_col_weight', type=int, default=3,
                    help='LDPC column weight')

    parser.add_argument('--iot_ldpc_max_iter', type=int, default=20,
                    help='LDPC decoding iterations')
    
    parser.add_argument(
        '--crop_scale_ratio',
        type=float,
        default=None,
        help='Ratio of image side length to keep before scaling back. E.g., 0.5 means 2x Zoom.'
    )
    parser.add_argument(
        '--composite_crop_jpeg',
        nargs=2,
        type=float,
        default=None,
        help='[crop_ratio, jpeg_quality]'
    )

    # IoT bitstream transmission attack
    parser.add_argument(
        '--iot_attack',
        action='store_true',
        help='Enable bitstream-level IoT transmission attack'
    )
    parser.add_argument(
        '--iot_jpeg_quality',
        type=int,
        default=50,
        help='JPEG quality used in IoT source coding'
    )
    parser.add_argument(
        '--iot_packet_bytes',
        type=int,
        default=1024,
        help='Packet size in bytes for IoT transmission'
    )
    parser.add_argument(
        '--iot_severity',
        type=str,
        default='moderate',
        choices=['mild', 'moderate', 'severe'],
        help='Burst channel severity'
    )
    parser.add_argument(
        '--iot_channel_mode',
        type=str,
        default='mixed',
        choices=['erasure', 'corruption', 'mixed'],
        help='Packet impairment type in IoT channel'
    )

    # 算法选择
    parser.add_argument(
        '--algo',
        default='holo',
        choices=['gs', 'holo', 'prc', 'gssync'],
        help='Algorithm type: gs, holo, prc, or gssync'
    )

    parser.add_argument('--run_name', default='test_run', type=str, help='Name of this experiment run')
    parser.add_argument('--contrast_factor', type=float, default=None)
    parser.add_argument('--vae_attack', action='store_true', help='Enable VAE Compression attack')
    parser.add_argument(
        '--vae_quality',
        type=int,
        default=3,
        help='Quality for VAE Compression (1-8). Lower is stronger attack. Default 3.'
    )

    args = parser.parse_args()

    # 路径拼接
    args.output_path = os.path.join(args.output_path, args.run_name)
    os.makedirs(args.output_path, exist_ok=True)

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)