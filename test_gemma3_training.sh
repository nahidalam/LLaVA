#!/bin/bash

# Minimal training test for Gemma3 vision encoder
# This script runs just a few training steps to verify the integration works

echo "=========================================="
echo "GEMMA3 VISION ENCODER - MINIMAL TRAINING TEST"
echo "=========================================="
echo ""
echo "This will run 10 training steps to verify the Gemma3 encoder works correctly."
echo "Check for:"
echo "  1. Model loads without errors"
echo "  2. Training starts and loss is computed"
echo "  3. No vision tower related errors"
echo ""

# Create a minimal test dataset if it doesn't exist
TEST_DATA_FILE="./test_gemma3_data.json"
if [ ! -f "$TEST_DATA_FILE" ]; then
    echo "Creating minimal test dataset..."
    cat > "$TEST_DATA_FILE" << 'EOF'
[
    {
        "id": "test_001",
        "image": "test_image.jpg",
        "conversations": [
            {"from": "human", "value": "What is in this image?"},
            {"from": "gpt", "value": "This is a test image for validating the Gemma3 vision encoder."}
        ]
    }
]
EOF
    echo "Test dataset created at $TEST_DATA_FILE"
fi

# Create a test image if it doesn't exist
TEST_IMAGE="./test_image.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    echo "Creating test image..."
    python -c "
from PIL import Image
import numpy as np
# Create a simple gradient image
img_array = np.zeros((224, 224, 3), dtype=np.uint8)
img_array[:, :, 0] = np.linspace(0, 255, 224).astype(np.uint8)[:, np.newaxis]
img_array[:, :, 1] = np.linspace(0, 255, 224).astype(np.uint8)[np.newaxis, :]
img = Image.fromarray(img_array)
img.save('$TEST_IMAGE')
print('Test image created')
"
fi

echo ""
echo "Starting minimal training test..."
echo "=========================================="

# Run training for just 10 steps
python llava/train/train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path "$TEST_DATA_FILE" \
    --image_folder ./ \
    --vision_tower hi-wesley/gemma3-vision-encoder \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./test_gemma3_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 512 \
    --lazy_preprocess True \
    --dataloader_num_workers 0 \
    --report_to none \
    --max_steps 10

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Training test completed without errors!"
    echo ""
    echo "The Gemma3 vision encoder is working correctly with LLaVA."
    echo "You can now run the full pretraining with:"
    echo "  bash scripts/v1_5/pretrain_llava_gemma3.sh"
else
    echo "❌ FAILED: Training test encountered errors."
    echo "Please check the error messages above."
fi
echo "=========================================="

# Cleanup test files (optional)
# rm -f "$TEST_DATA_FILE" "$TEST_IMAGE"
# rm -rf ./test_gemma3_output