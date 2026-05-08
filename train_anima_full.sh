

# ===== Paths =====
SD_SCRIPTS_DIR="/home/gazai/joe_workspace/AI-Server/method/lora_train/sd-scripts-wrap/sd-scripts"
WORK_DIR="/home/gazai/joe_workspace/musubi-tuner-workspace/Anima_GAZAI_Anime_Style-v0.4-5e-8"
TRAIN_DIR="$WORK_DIR/train"
DATASET_CONFIG="$WORK_DIR/anima_dataset.toml"
OUTPUT_DIR="$WORK_DIR/output"
LOG_DIR="$WORK_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ===== Environment =====
source "/home/gazai/joe_workspace/AI-Server/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=1

# ===== WandB =====
WANDB_API_KEY="1cf44800290016a7965c21c46796365b3fcb07eb"
WANDB_RUN_NAME="anima_GAZAI_anime_style-v0.3-5e-8"
export WANDB_API_KEY

# ===== Model paths =====
PRETRAINED_MODEL_NAME_OR_PATH="/home/gazai/models/checkpoints/anima-preview3-base.safetensors"
QWEN3_PATH="/home/gazai/models/text_encoders/qwen_3_06b_base.safetensors"
VAE_PATH="/home/gazai/models/vae/qwen_image_vae_train.safetensors"

[[ -f "$PRETRAINED_MODEL_NAME_OR_PATH" ]] || { echo "Missing model: $PRETRAINED_MODEL_NAME_OR_PATH"; exit 1; }
[[ -e "$QWEN3_PATH" ]] || { echo "Missing Qwen3 path: $QWEN3_PATH"; echo "Please set QWEN3_PATH to the Anima-required Qwen3-0.6B model path."; exit 1; }
[[ -f "$VAE_PATH" ]] || { echo "Missing VAE: $VAE_PATH"; exit 1; }
[[ -f "$DATASET_CONFIG" ]] || { echo "Missing dataset config: $DATASET_CONFIG"; exit 1; }

# ===== Training params (official-style baseline for Anima full finetune) =====
MAX_TRAIN_EPOCHS="20"
LEARNING_RATE="5e-8"
TRAIN_BATCH_SIZE="16"
GRAD_ACCUM_STEPS="1"
SAVE_EVERY_N_EPOCHS="2"
OUTPUT_NAME=$WANDB_RUN_NAME
TIMESTEP_SAMPLING="sigmoid"
SIGMOID_SCALE="1.0"
MAX_DATA_LOADER_N_WORKERS="1"

# ===== In-training sample test options =====
SAMPLE_PROMPTS="$WORK_DIR/sample_prompts.txt"
SAMPLE_EVERY_N_EPOCHS="1"

CMD=(
  accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 "$SD_SCRIPTS_DIR/anima_train.py"
  --pretrained_model_name_or_path="$PRETRAINED_MODEL_NAME_OR_PATH"
  --qwen3="$QWEN3_PATH"
  --vae="$VAE_PATH"
  --dataset_config="$DATASET_CONFIG"
  --output_dir="$OUTPUT_DIR"
  --output_name="$OUTPUT_NAME"
  --save_model_as safetensors
  --save_precision bf16
  --max_train_epochs "$MAX_TRAIN_EPOCHS"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
  --learning_rate "$LEARNING_RATE"
  --llm_adapter_lr 0 
  --optimizer_type AdamW
  --optimizer_args weight_decay=0.01 betas=0.9,0.95
  --timestep_sampling "$TIMESTEP_SAMPLING"
  --sigmoid_scale "$SIGMOID_SCALE"
  --mixed_precision bf16
  --gradient_checkpointing
  --cache_latents
  --cache_latents_to_disk
  --vae_batch_size 32
  --max_data_loader_n_workers "$MAX_DATA_LOADER_N_WORKERS"
  --persistent_data_loader_workers
  --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
  --save_state
  --save_state_on_train_end
  --logging_dir "$LOG_DIR"
  --log_with wandb
  --shuffle_caption
  --keep_tokens=1
  --resolution=1024
  --enable_bucket
  --min_bucket_reso=700
  --max_bucket_reso=1300
  --bucket_reso_steps=64
  --caption_tag_dropout_rate=0.1
  --wandb_run_name "$WANDB_RUN_NAME"
  --sample_prompts "$SAMPLE_PROMPTS"
  --sample_sampler "k_euler_a"
  --sample_every_n_epochs "$SAMPLE_EVERY_N_EPOCHS"
  --attn_mode sdpa
)


# ===== Run training in background with nohup =====
nohup "${CMD[@]}" > "$LOG_DIR/train_output.log" 2>&1 &
# "${CMD[@]}"


# nvidia-smi -l 1

# sudo kill $(sudo pgrep -f "anima_train.py")
# sudo kill $(sudo pgrep -f "wandb_run_name anima_GAZAI_anime_style-v0.3-5e-8")