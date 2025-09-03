# Last modified: 2025-09-03 15:19:52
appVersion = "0.0.7"
"""
Gradio app for image description using FastVLM-1.5B.
"""

import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Model identifier and special token configuration.  The IMAGE_TOKEN_INDEX is
# inserted into the text sequence to indicate where the vision features
# belong.  It must match the value expected by the underlying model code.
# MID = "apple/FastVLM-1.5B"
MID = "apple/FastVLM-7B"
# MODEL_LOCAL_DIR = os.path.join("models", "FastVLM-1.5B")
MODEL_LOCAL_DIR = os.path.join("models", "FastVLM-7B")
IMAGE_TOKEN_INDEX = -200


def _resolve_local_model_dir(base_dir: str) -> str | None:
    """Find a directory under base_dir that contains a config.json.

    Returns the directory path if found; otherwise None.
    """
    cfg = os.path.join(base_dir, "config.json")
    if os.path.isfile(cfg):
        return base_dir
    # Walk a few levels to accommodate huggingface_hub snapshot layout
    for root, _dirs, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    return None


def load_model():
    """Load the tokenizer and language model once.

    Returns a tuple of (tokenizer, model) on the appropriate device.
    """
    # Determine the appropriate dtype based on hardware availability.  Use
    # float16 on GPU for efficiency; fall back to float32 on CPU.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # Prefer a local copy in models/FastVLM-1.5B if available. Otherwise
    # load from Hugging Face, caching into the local models directory.
    local_source = (
        _resolve_local_model_dir(MODEL_LOCAL_DIR)
        if os.path.isdir(MODEL_LOCAL_DIR)
        else None
    )
    source_path = local_source or MID
    tokenizer = AutoTokenizer.from_pretrained(
        source_path,
        trust_remote_code=True,
        cache_dir=MODEL_LOCAL_DIR if local_source is None else None,
        local_files_only=local_source is not None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        source_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=MODEL_LOCAL_DIR if local_source is None else None,
        local_files_only=local_source is not None,
    )
    # Ensure pad/eos token IDs exist so generation can terminate cleanly
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model


# Load tokenizer and model at import time to avoid repeated downloads.
TOKENIZER, MODEL = load_model()


def describe_image(img: Image.Image) -> str:
    """Generate a detailed description for the uploaded image.

    Args:
        img: A PIL Image object.

    Returns:
        A string containing the model's description of the image.
    """
    # Build the chat template. The model expects a prompt containing
    # `<image>` where the vision features will be inserted. Request a
    # strict JSON response including an object list. The response must be
    # enclosed in a ```json code block with no extra text.
    structured_instruction = (
        "<image>\nAnalyze the image and respond ONLY with a JSON object, "
        "enclosed within a ```json code block. Do NOT include any text "
        "outside of the JSON. The JSON must have the following keys: "
        "'who', 'gender', 'age', 'ethnicity', 'what', 'when', 'where', "
        "'why', 'how', 'detailed_summary', 'additional', 'confidence', "
        "and 'object_list' (an array).\n"
        "- Use appearance-based, non-judgmental phrasing (e.g., 'appears to be').\n"
        "- If uncertain about sensitive attributes (gender/age/ethnicity), use 'Unknown'.\n"
        "- confidence is a number 0-100.\n"
        "- object_list: identify all visible objects. Each item is an object with: "
        "  name (string), description (string), x (number), y (number).\n"
        "  x and y are relative coordinates of the object's center in the range [0,1], "
        "  where (0,0) is top-left and (1,1) is bottom-right.\n"
        "Output format example (you MUST follow the same structure and key names):\n"
        "```json\n"
        "{\n"
        "  \"who\": \"Unknown\",\n"
        "  \"gender\": \"Unknown\",\n"
        "  \"age\": \"Unknown\",\n"
        "  \"ethnicity\": \"Unknown\",\n"
        "  \"what\": \"\",\n"
        "  \"when\": \"\",\n"
        "  \"where\": \"\",\n"
        "  \"why\": \"\",\n"
        "  \"how\": \"\",\n"
        "  \"detailed_summary\": \"\",\n"
        "  \"additional\": \"\",\n"
        "  \"confidence\": 75,\n"
        "  \"object_list\": [\n"
        "    { \"name\": \"\", \"description\": \"\", \"x\": 0.5, \"y\": 0.5 }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Return ONLY the code block above with your filled-in values."
    )
    messages = [
        {"role": "user", "content": structured_instruction}
    ]
    rendered = TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # Split the template around the <image> token.  We avoid adding special
    # tokens here so that we can insert the IMAGE_TOKEN_INDEX directly.
    pre, post = rendered.split("<image>", 1)
    pre_ids = TOKENIZER(
        pre, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(MODEL.device)
    post_ids = TOKENIZER(
        post, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(MODEL.device)
    # Create a tensor for the image token and concatenate with pre/post text.
    img_tok = torch.tensor(
        [[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype, device=MODEL.device
    )
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
    # Build the attention mask (all ones) matching the input length.
    attention_mask = torch.ones_like(input_ids, device=MODEL.device)
    # Ensure the image is a PIL Image and preprocess it with the model's
    # vision tower.  The processor handles resizing, normalization, etc.
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    px = (
        MODEL.get_vision_tower()
        .image_processor(images=img, return_tensors="pt")["pixel_values"]
        .to(MODEL.device, dtype=MODEL.dtype)
    )
    # Generate the description. Choose a safe token budget to avoid
    # truncation by context length, and decode only new tokens.
    input_len = input_ids.shape[1]
    max_ctx = getattr(MODEL.config, "tokenizer_model_max_length", None)
    if not isinstance(max_ctx, int) or max_ctx <= 0 or max_ctx > 2_000_000:
        max_ctx = 8192
    room = max(16, max_ctx - input_len - 8)
    gen_max = int(min(512, room))

    with torch.no_grad():
        output_ids = MODEL.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=gen_max,
            eos_token_id=MODEL.config.eos_token_id,
            pad_token_id=MODEL.config.pad_token_id,
        )
    # Some custom generate implementations may NOT include the prompt in
    # the returned sequences. If the output doesn't start with the input,
    # decode the whole thing to avoid chopping off the beginning.
    seq = output_ids[0]
    if seq.shape[0] >= input_len and torch.equal(seq[:input_len], input_ids[0]):
        gen_only = seq[input_len:]
    else:
        gen_only = seq
    text = TOKENIZER.decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return text


def main():
    """Create and launch the Gradio interface."""
    interface = gr.Interface(
        fn=describe_image,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="Image Description using FastVLM-1.5B",
        description=(
            "Upload an image and receive a detailed description generated by the "
            "FastVLM-1.5B model. The model uses both vision and language "
            "capabilities to produce coherent, descriptive text."
        ),
    )
    # Launch the app. Setting share=False by default; set to True if you need
    # a public link via Gradio's servers.
    interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
