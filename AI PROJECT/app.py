import os
from flask import Flask, render_template, request, send_file
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import torch

app = Flask(__name__)

# Ensure output directory exists
os.makedirs("static/output", exist_ok=True)

# Text generation pipeline (using GPT-2)
text_generator = pipeline("text-generation", model="gpt2")

# Image generation pipeline (using Stable Diffusion)
stable_diffusion = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32  # Use float32 for CPU
)
stable_diffusion.to("cpu")  # Explicitly move the model to CPU


# Function to generate dialogue for panels

def generate_dialogue(prompt, num_panels=1, max_new_tokens=150):
    # Initialize the text generation model
    generator = pipeline(
        "text-generation", 
        model="gpt2",
        device=-1  # Force CPU usage
    )
    
    dialogues = []
    for _ in range(num_panels):  # Generate dialogues for each panel
        dialogue = generator(
            prompt, 
            max_new_tokens=max_new_tokens,  # Control the length of new tokens only
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True
        )[0]["generated_text"]
        dialogues.append(dialogue)
    
    return dialogues




# Function to generate an image
def generate_image(prompt, save_path):
    image = stable_diffusion(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(save_path)
    return save_path


# Function to add speech bubbles
def add_speech_bubble(image_path, text, bubble_position=(50, 50), save_path="comic_with_bubble.png", max_width=300):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use Arial font
        font = ImageFont.truetype("arial.ttf", size=20)
    except OSError:
        # Fallback to default font if Arial is not available
        font = ImageFont.load_default()

    # Split the text into lines that fit within the bubble width
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = f"{current_line} {word}".strip()
        # Use textlength instead of textsize
        test_width = draw.textlength(test_line, font=font)
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:  # Append the last line
        lines.append(current_line)

    # Calculate bubble dimensions using textbbox
    # Get the bounding box for a sample text to calculate line height
    sample_bbox = draw.textbbox((0, 0), "hg", font=font)
    line_height = sample_bbox[3] - sample_bbox[1]  # Height of one line
    
    # Calculate maximum width of all lines
    max_line_width = max(draw.textlength(line, font=font) for line in lines)
    bubble_width = max_line_width + 20  # Add padding
    bubble_height = len(lines) * line_height + 20  # Total height including padding

    # Get bubble position
    bubble_x, bubble_y = bubble_position

    # Draw the bubble background
    draw.rectangle(
        [(bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height)],
        fill="white", outline="black", width=2
    )

    # Draw each line of text
    y_offset = bubble_y + 10  # Start padding
    for line in lines:
        # Calculate text bounding box for proper vertical centering
        bbox = draw.textbbox((bubble_x + 10, y_offset), line, font=font)
        draw.text((bubble_x + 10, y_offset), line, fill="black", font=font)
        y_offset += line_height  # Move to the next line

    # Save the updated image
    image.save(save_path)
    return save_path


# Function to arrange panels into a comic strip
def create_comic_strip(panel_paths, save_path):
    images = [Image.open(panel) for panel in panel_paths]
    widths, heights = zip(*(img.size for img in images))
    total_width = max(widths)
    total_height = sum(heights)

    comic_strip = Image.new("RGB", (total_width, total_height), "white")
    y_offset = 0
    for img in images:
        comic_strip.paste(img, (0, y_offset))
        y_offset += img.height

    comic_strip.save(save_path)
    return save_path


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_comic():
    story = request.form.get("story")
    num_panels = 2

    try:
        # Step 1: Generate dialogues for panels
        dialogues = generate_dialogue(story, num_panels=num_panels)

        # Step 2: Generate images and add speech bubbles
        panel_paths = []
        for i, dialogue in enumerate(dialogues):
            prompt = f"{story} - {dialogue}"
            image_path = f"static/output/panel_{i}.png"
            bubble_image_path = f"static/output/panel_with_bubble_{i}.png"

            # Generate image
            generate_image(prompt, image_path)

            # Add speech bubble
            add_speech_bubble(image_path, dialogue, save_path=bubble_image_path)
            panel_paths.append(bubble_image_path)

        # Step 3: Arrange panels into a comic strip
        comic_path = "static/output/final_comic.png"
        create_comic_strip(panel_paths, comic_path)

        return send_file(comic_path, mimetype="image/png", as_attachment=True)

    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
