#!/bin/bash

REPO="Inevitablevalor/MindCube"
TARGET_DIR="dataset/mindcube"
ZIP_NAME="/tmp/mindcube_data.zip"
HF_URL="https://huggingface.co/datasets/$REPO/resolve/main/data.zip"

echo "Downloading MindCube dataset from Hugging Face..."

# Method 1: Direct curl download (no dependencies, works for public datasets)
echo "Attempting direct download via curl..."
curl -L --fail --progress-bar "$HF_URL" -o "$ZIP_NAME"

if [ $? -ne 0 ] || [ ! -s "$ZIP_NAME" ]; then
    echo "Direct download failed. Trying huggingface-cli..."
    rm -f "$ZIP_NAME"

    # Method 2: huggingface-cli (handles auth for private repos)
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing huggingface-hub..."
        pip install -q huggingface-hub
    fi

    huggingface-cli download "$REPO" data.zip \
        --local-dir /tmp/hf_mindcube \
        --repo-type dataset

    if [ -f "/tmp/hf_mindcube/data.zip" ]; then
        mv /tmp/hf_mindcube/data.zip "$ZIP_NAME"
        rm -rf /tmp/hf_mindcube
    else
        rm -rf /tmp/hf_mindcube
        echo ""
        echo "❌ All download methods failed. Possible reasons:"
        echo "   • Dataset is private — run: huggingface-cli login"
        echo "   • Network connectivity issue"
        echo "   • Verify repo exists: https://huggingface.co/datasets/$REPO"
        exit 1
    fi
fi

# Extract
echo "Extracting dataset to $TARGET_DIR ..."
mkdir -p "$TARGET_DIR"
unzip -q -o "$ZIP_NAME" -d "$TARGET_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Extraction failed."
    rm -f "$ZIP_NAME"
    exit 1
fi

rm -f "$ZIP_NAME"
echo "✓ Dataset ready at ./$TARGET_DIR"
echo ""
ls "$TARGET_DIR/"
