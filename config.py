NUM_NEGATIVE_IMAGES = 500
NUM_KL_PROMPTS = 20

HF_TOKEN = ""
if not HF_TOKEN:
    raise Exception("Huggingface token is required. Please add your HG token to config.py")