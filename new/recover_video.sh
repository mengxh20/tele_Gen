python new/visualize_helios_stage1_dataset.py /gemini/platform/public/luojx/team/mengxh/codes/Helios/example/toy_data/latents_short/toy_data \
    --base-model-path /gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/Helios-Base \
    --source-video-dir /gemini/platform/public/luojx/team/mengxh/codes/Helios/example/toy_data/videos \
    --filter-json /gemini/platform/public/luojx/team/mengxh/codes/Helios/example/toy_data/toy_filter.json \
    --decode-sections all \
    --compute-metrics \
    --save-video \
    --fps 30
