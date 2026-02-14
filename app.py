import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO
import pickle
import os

# Page config
st.set_page_config(
    page_title="🖼️ Neural Storyteller",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern gradient UI inspired by your CV project
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 60%, #0b1224 100%);
    }

    /* Main card */
    .main .block-container {
        background: rgba(18, 27, 48, 0.98);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    /* Title */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.7rem;
        font-weight: 800;
        color: #f8fafc;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.35);
    }

    .subtitle {
        text-align: center;
        color: #e5e7eb;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
        font-weight: 500;
    }

    /* Caption card */
    .caption-box {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 50%, #f97316 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        color: #111827;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    .caption-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #111827;
        opacity: 0.8;
        margin-bottom: 0.3rem;
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #38bdf8;
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        color: #e5e7eb;
        font-size: 0.9rem;
        margin-top: 0.8rem;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #0b1224 0%, #0f172a 100%);
        padding: 1.6rem 1.4rem;
        border-radius: 16px;
        border: 2px dashed rgba(248, 191, 36, 0.7);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    }

    section[data-testid="stFileUploader"] label {
        color: #f8fafc !important;
        font-weight: 600;
    }

    section[data-testid="stFileUploader"] small {
        color: #9ca3af !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f97316 0%, #f59e0b 45%, #fbbf24 100%);
        color: #111827;
        border: none;
        padding: 0.7rem 2.4rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.35);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0b1120 100%);
        border-right: 1px solid rgba(30, 64, 175, 0.6);
    }

    section[data-testid="stSidebar"] label {
        color: #e5e7eb !important;
        font-weight: 600;
    }

    /* Metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #020617 0%, #0b1120 100%);
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-weight: 500;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e5e7eb !important;
        font-size: 1.4rem !important;
        font-weight: 700;
    }

    /* Hide default header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Model classes
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc = nn.Linear(2048, embed_size)
    
    def forward(self, x):
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, captions, hidden):
        embeds = self.embedding(captions)
        outputs, hidden_out = self.lstm(embeds, hidden)
        return self.fc(outputs), hidden_out

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, imgs, captions):
        enc_out = self.encoder(imgs)
        hidden = (enc_out.unsqueeze(0), torch.zeros_like(enc_out).unsqueeze(0))
        outputs, _ = self.decoder(captions[:, :-1], hidden)
        return outputs

# Load model and vocab
@st.cache_resource
def load_model_and_vocab():
    # Load vocabulary
    if not os.path.exists('vocab.pkl'):
        raise FileNotFoundError("vocab.pkl not found in the current directory.")

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Prefer precomputed inverse vocab if available
    if os.path.exists('inv_vocab.pkl'):
        with open('inv_vocab.pkl', 'rb') as f:
            inv_vocab = pickle.load(f)
    else:
        inv_vocab = {idx: word for word, idx in vocab.items()}

    vocab_size = len(vocab)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioningModel(
        Encoder(512),
        Decoder(512, 512, vocab_size)
    ).to(device)

    # Load weights
    if not os.path.exists('image_captioning_model.pth'):
        raise FileNotFoundError("image_captioning_model.pth not found in the current directory.")

    model.load_state_dict(torch.load('image_captioning_model.pth', map_location=device))
    model.eval()

    return model, vocab, inv_vocab, device

# Feature extraction
@st.cache_resource
def load_feature_extractor():
    from torchvision import models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()
    return resnet, device

def extract_features(image, resnet, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor).view(1, -1)
    return features.squeeze(0)

# Inference functions
def greedy_search(model, img_feat, vocab, inv_vocab, device, max_len=20):
    model.eval()
    with torch.no_grad():
        enc = model.encoder(img_feat.unsqueeze(0).to(device))
        hidden = (enc.unsqueeze(0), torch.zeros_like(enc).unsqueeze(0))
        
        word = torch.tensor([[vocab["<start>"]]]).to(device)
        result = []
        
        for _ in range(max_len):
            # Only feed the last token each step to avoid reprocessing the full sequence
            out, hidden = model.decoder(word[:, -1:].contiguous(), hidden)
            pred = out.argmax(-1)[:, -1]
            
            if pred.item() == vocab["<end>"]:
                break
            
            result.append(inv_vocab[pred.item()])
            word = torch.cat([word, pred.unsqueeze(1)], dim=1)
    
    return " ".join(result)

def beam_search(model, img_feat, vocab, inv_vocab, device, beam_width=3, max_len=20):
    """Beam search with proper hidden propagation and length-normalized scores."""
    model.eval()
    with torch.no_grad():
        enc = model.encoder(img_feat.unsqueeze(0).to(device))
        h0 = enc.unsqueeze(0)
        c0 = torch.zeros_like(h0)

        # Each beam: (sequence, score, hidden_state)
        beams = [([vocab["<start>"]], 0.0, (h0, c0))]

        for _ in range(max_len):
            candidates = []

            for seq, score, hidden in beams:
                if seq[-1] == vocab["<end>"]:
                    candidates.append((seq, score, hidden))
                    continue

                # Only feed the last token for the next step
                last_token = torch.tensor([[seq[-1]]], device=device)
                out, new_hidden = model.decoder(last_token, hidden)
                probs = torch.log_softmax(out[:, -1, :], dim=-1)

                topk_probs, topk_indices = torch.topk(probs, beam_width)

                for i in range(beam_width):
                    token = topk_indices[0, i].item()
                    new_seq = seq + [token]
                    new_score = score + topk_probs[0, i].item()

                    # Length-normalized score (avoid overly short/odd beams)
                    norm_score = new_score / len(new_seq)
                    candidates.append((new_seq, norm_score, new_hidden))

            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            if all(seq[-1] == vocab["<end>"] for seq, _, _ in beams):
                break

        best_seq = beams[0][0]
        result = [inv_vocab[idx] for idx in best_seq if idx not in [vocab["<start>"], vocab["<end>"], vocab["<pad>"]]]

    return " ".join(result)


# Main app
def main():
    # Hero header
    st.markdown('<h1 class="main-title">Neural Storyteller</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Image caption generation using ResNet50 + Seq2Seq (Encoder–Decoder with LSTM)</p>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Caption Settings")
        st.markdown("---")

        method = st.radio(
            "Generation strategy",
            ["Beam Search (better quality)", "Greedy Search (fast)"],
            index=0,
            help="Beam search explores multiple caption paths; greedy picks the best word each step.",
        )

        max_len = st.slider(
            "Maximum caption length",
            min_value=8,
            max_value=25,
            value=20,
            step=1,
        )

        beam_width = st.slider(
            "Beam width (for beam search)",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="info-box">
                <strong>Tip:</strong> Beam search with width 3–5 usually balances quality and speed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load models
    try:
        with st.spinner("🔄 Loading captioning model and feature extractor..."):
            model, vocab, inv_vocab, device = load_model_and_vocab()
            resnet, feat_device = load_feature_extractor()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.info("Ensure 'image_captioning_model.pth' and 'vocab.pkl' (and optional 'inv_vocab.pkl') are beside app.py.")
        return

    st.markdown("### 📤 Upload an image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload any image and the model will describe it.",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to generate a story-like caption.")
        st.markdown(
            """
            <div class="info-box">
                <strong>Behind the scenes:</strong> We extract visual features with ResNet50, then a Seq2Seq
                decoder generates the caption word-by-word.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Display uploaded image and metadata
    try:
        file_bytes = uploaded_file.read()
        image = Image.open(BytesIO(file_bytes)).convert("RGBA").convert("RGB")
    except Exception as e:
        st.error(f"⚠️ Could not read the image file: {uploaded_file.name}\n{e}")
        st.info("Try re-uploading a standard JPG/PNG. If it still fails, re-save the image locally and upload again.")
        return
    col_img, col_meta = st.columns([2.2, 1.3])

    with col_img:
        st.image(image, use_container_width=True, caption="Uploaded image")

    with col_meta:
        st.markdown("#### 📂 File details")
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

    generate = st.button("✨ Generate Caption")

    if not generate:
        return

    with st.spinner("🎨 Crafting a caption from the image..."):
        # Extract features
        features = extract_features(image, resnet, feat_device)

        # Generate caption
        if method.startswith("Beam"):
            caption = beam_search(
                model,
                features,
                vocab,
                inv_vocab,
                device,
                beam_width=beam_width,
                max_len=max_len,
            )
        else:
            caption = greedy_search(model, features, vocab, inv_vocab, device, max_len=max_len)

    # Show caption and simple metrics
    st.markdown(
        f"""
        <div class="caption-box">
            <div class="caption-label">Generated caption</div>
            <div>📝 {caption.capitalize()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tokens = caption.split()
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Tokens in caption", len(tokens))
    with col_m2:
        st.metric("Generation mode", "Beam" if method.startswith("Beam") else "Greedy")
    with col_m3:
        st.metric("Max length used", max_len)

    st.markdown(
        """
        <div class="info-box">
            <strong>Note:</strong> This baseline model was trained on Flickr30k with a simple Encoder–Decoder.
            Captions may be approximate but usually capture the main objects and actions in the scene.
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
