call venv/Scripts/activate
accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train_network.py" ^
    --pretrained_model_name_or_path="C:/Users/endma/Desktop/stable-diffusion-webui/models/Stable-diffusion/recommend/sd_xl_base_1.0.safetensors" ^
    --train_data_dir="D:/AIGC/dataset/oneImageTest" ^
    --output_dir="D:/AIGC/model/oneImageTest" ^
    --logging_dir="D:/AIGC/model/oneImageTest" ^
    --training_comment="trigger: oneImageTest" --save_model_as=safetensors ^
    --train_batch_size="2" --max_train_steps="500" --save_every_n_epochs="1" ^
    --resolution="1024,1024" --network_alpha="1" ^
    --network_module=networks.lora --network_dim=4 ^
    --output_name="oneImageTest" ^
    --cache_text_encoder_outputs --learning_rate="1e-4 "  ^
    --mixed_precision="bf16" --save_precision="bf16" --caption_extension=".txt" --cache_latents ^
    --cache_latents_to_disk --optimizer_type="AdamW8bit" --max_data_loader_n_workers="1" --gradient_checkpointing --xformers ^
    --log_prefix=xl-loha --network_train_unet_only ^
    --persistent_data_loader_workers --lowram  --log_with="wandb"
    @REM --log_with="wandb"


    @REM 
    @REM --max_timestep 40
