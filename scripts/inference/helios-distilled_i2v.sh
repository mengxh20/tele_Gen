CUDA_VISIBLE_DEVICES=6,7 python infer_helios.py \
    --base_model_path "/gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/HeliosDistillede" \
    --transformer_path "/gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/HeliosDistillede" \
    --sample_type "i2v" \
    --image_path "example/wave.jpg" \
    --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and respect for nature’s might." \
    --num_frames 240 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --enable_compile \
    --output_folder "/gemini/platform/public/luojx/team/mengxh/codes/Helios/output_helios/helios-distilled"


    # --pyramid_num_inference_steps_list 1 1 1 \