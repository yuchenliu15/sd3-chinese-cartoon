export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="../training_data"
export OUTPUT_DIR="../finetuned_output"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="cartoon characters and machinery setups" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-3 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="two pandas using python to crunch numbers" \
  --validation_epochs=50 \
  --seed="0" \
