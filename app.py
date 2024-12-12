import os
import random
import torch
import gradio as gr
import logging
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from omegaconf import OmegaConf
from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
from memo.pipelines.video_pipeline import VideoPipeline
from memo.utils.audio_utils import extract_audio_emotion_labels, preprocess_audio, resample_audio
from memo.utils.vision_utils import preprocess_image, tensor_to_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memo")

def load_models(config_path="configs/inference.yaml"):
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(config.weight_dtype, torch.float32)

    logger.info("Loading models")
    vae = AutoencoderKL.from_pretrained(config.vae).to(device=device, dtype=weight_dtype)
    reference_net = UNet2DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="reference_net", use_safetensors=True
    ).to(device=device, dtype=weight_dtype)
    diffusion_net = UNet3DConditionModel.from_pretrained(
        config.model_name_or_path, subfolder="diffusion_net", use_safetensors=True
    ).to(device=device, dtype=weight_dtype)
    image_proj = ImageProjModel.from_pretrained(
        config.model_name_or_path, subfolder="image_proj", use_safetensors=True
    ).to(device=device, dtype=weight_dtype)
    audio_proj = AudioProjModel.from_pretrained(
        config.model_name_or_path, subfolder="audio_proj", use_safetensors=True
    ).to(device=device, dtype=weight_dtype)

    vae.requires_grad_(False).eval()
    reference_net.requires_grad_(False).eval()
    diffusion_net.requires_grad_(False).eval()
    image_proj.requires_grad_(False).eval()
    audio_proj.requires_grad_(False).eval()

    if config.enable_xformers_memory_efficient_attention:
        reference_net.enable_xformers_memory_efficient_attention()
        diffusion_net.enable_xformers_memory_efficient_attention()

    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline = VideoPipeline(
        vae=vae,
        reference_net=reference_net,
        diffusion_net=diffusion_net,
        scheduler=noise_scheduler,
        image_proj=image_proj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    face_analysis = os.path.join(config.misc_model_dir, "misc/face_analysis")
    vocal_separator = os.path.join(config.misc_model_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")

    output_dir = getattr(config, "output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    return pipeline, audio_proj, config, face_analysis, vocal_separator, output_dir, device

def generate(input_video, input_audio, seed):
    pipeline, audio_proj, config, face_analysis, vocal_separator, output_dir, device = load_models()

    resolution = config.resolution
    fps = config.fps
    num_generated_frames_per_clip = config.num_generated_frames_per_clip
    num_init_past_frames = config.num_init_past_frames
    num_past_frames = config.num_past_frames
    inference_steps = config.inference_steps
    cfg_scale = config.cfg_scale

    if seed == 0:
        seed = random.randint(0, 18446744073709551615)

    generator = torch.manual_seed(seed)
    img_size = (resolution, resolution)
    pixel_values, face_emb = preprocess_image(
        face_analysis_model=face_analysis,
        image_path=input_video,
        image_size=resolution,
    )
    pixel_values = pixel_values.to(device=device)

    cache_dir = os.path.join(output_dir, "audio_preprocess")
    os.makedirs(cache_dir, exist_ok=True)
    input_audio = resample_audio(
        input_audio,
        os.path.join(cache_dir, f"{os.path.basename(input_audio).split('.')[0]}-16k.wav"),
    )

    audio_emb, audio_length = preprocess_audio(
        wav_path=input_audio,
        num_generated_frames_per_clip=num_generated_frames_per_clip,
        fps=fps,
        wav2vec_model=config.wav2vec,
        vocal_separator_model=vocal_separator,
        cache_dir=cache_dir,
        device=device,
    )

    audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
        model=config.model_name_or_path,
        wav_path=input_audio,
        emotion2vec_model=config.emotion2vec,
        audio_length=audio_length,
        device=device,
    )

    video_frames = []
    num_clips = audio_emb.shape[0] // num_generated_frames_per_clip
    for t in tqdm(range(num_clips), desc="Generating video clips"):
        if len(video_frames) == 0:
            past_frames = pixel_values.repeat(num_init_past_frames, 1, 1, 1).to(device=device)
        else:
            past_frames = video_frames[-1][0].permute(1, 0, 2, 3)[-num_past_frames:].to(device=device)
        past_frames = past_frames.to(dtype=pixel_values.dtype)
        pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0).unsqueeze(0)

        audio_tensor = (
            audio_emb[
                t * num_generated_frames_per_clip : min(
                    (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
                )
            ]
            .unsqueeze(0)
            .to(device=audio_proj.device, dtype=audio_proj.dtype)
        )
        audio_tensor = audio_proj(audio_tensor)

        audio_emotion_tensor = audio_emotion[
            t * num_generated_frames_per_clip : min(
                (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
            )
        ]

        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            audio_emotion=audio_emotion_tensor,
            emotion_class_num=num_emotion_classes,
            face_emb=face_emb,
            width=img_size[0],
            height=img_size[1],
            video_length=num_generated_frames_per_clip,
            num_inference_steps=inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
        )

        video_frames.append(pipeline_output.videos)

    video_frames = torch.cat(video_frames, dim=2).squeeze(0)[:, :audio_length]
    output_video_path = os.path.join(output_dir, f"memo_output_{seed}.mp4")
    tensor_to_video(video_frames, output_video_path, input_audio, fps=fps)
    return output_video_path

with gr.Blocks(css=".gradio-container {max-width: 1080px !important}", analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            input_video = gr.Image(label="Upload Input Image", type="filepath")
            input_audio = gr.Audio(label="Upload Input Audio", type="filepath")
            seed = gr.Number(label="Seed (0 for Random)", value=0, precision=0)
        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            generate_button = gr.Button("Generate")

    generate_button.click(
        fn=generate,
        inputs=[input_video, input_audio, seed],
        outputs=[video_output],
    )

demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0', server_port=None)
