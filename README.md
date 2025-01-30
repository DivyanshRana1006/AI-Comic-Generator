# 🎨 AI-Powered Comic Book Generator 📖✨

Turn your stories into stunning AI-generated comics! This project uses **GPT-2 for text generation** and **Stable Diffusion for image generation** to create visually appealing comic strips with automatic speech bubbles. 🚀

## 📌 Features
✅ **Text-to-Dialogue Generation** – AI generates comic dialogues from input stories.  
✅ **AI Image Generation** – Uses Stable Diffusion to create visuals based on dialogue.  
✅ **Speech Bubble Integration** – Automatically adds dialogue to images with proper formatting.  
✅ **Comic Panel Arrangement** – Combines generated images into a structured comic strip.  
✅ **User-Friendly Web Interface** – Built using Flask for seamless interaction.  

## 🛠️ Tech Stack
- **Backend**: Flask, Python  
- **AI Models**: GPT-2 (Text Generation), Stable Diffusion (Image Generation)  
- **Front-End**: HTML, CSS, JavaScript  
- **Libraries**: PIL (for image processing), Hugging Face Transformers, Diffusers  

## 🚀 How It Works
1️⃣ Enter a short story or idea.  
2️⃣ AI generates dialogues for the comic.  
3️⃣ Stable Diffusion creates images for each panel.  
4️⃣ Speech bubbles are added to the images.  
5️⃣ The final comic strip is generated and ready to download!  

## 🖼️ Screenshots
Below are some images showcasing the project:

![Comic Generator UI]  
*User-friendly interface for inputting stories and generating comics.*  

![Generated Comic Panels] 
*AI-generated comic strip with speech bubbles.*  

## 🏃‍♂️ Run Locally
```bash
# Clone the repository
git clone https://github.com/your-username/comic-generator.git

# Navigate to the project directory
cd comic-generator

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
