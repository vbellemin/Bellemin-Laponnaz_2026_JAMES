#!/bin/bash

# ====================================================================================
# Script to download datasets on SEANOE, and place them in corresponding directories
# ====================================================================================

set -e  # stop script if any command fails

echo "----------------------------------------"
echo "Starting dataset download and extraction"
echo "----------------------------------------"

# Function to download, move, extract, and clean
process_tar () {

    URL=$1
    DEST=$2

    FILE=$(basename "$URL")

    echo ""
    echo "----------------------------------------"
    echo "Processing file: $FILE"
    echo "Source URL: $URL"
    echo "Destination: $DEST"
    echo "----------------------------------------"

    # Create destination directory if it doesn't exist
    echo "Creating destination directory..."
    mkdir -p "$DEST"

    # Download file
    echo "Downloading archive..."
    wget -c "$URL"

    # Move file to destination
    echo "Moving archive to destination folder..."
    mv "$FILE" "$DEST/"

    # Extract archive
    echo "Extracting archive..."
    tar -xf "$DEST/$FILE" -C "$DEST"

    # Delete tar file
    echo "Removing original archive..."
    rm "$DEST/$FILE"

    echo "Finished processing $FILE"
}

# ==========================================================
# Process each dataset
# ==========================================================

process_tar "https://www.seanoe.org/data/00966/107806/data/121201.tar" "./data/mapping_outputs/config_QGSW"

process_tar "https://www.seanoe.org/data/00966/107806/data/121202.tar" "./data/mapping_outputs/config_QG"

process_tar "https://www.seanoe.org/data/00966/107806/data/121204.tar" "./data/mapping_outputs/config_QGSW_notime"

process_tar "https://www.seanoe.org/data/00966/107806/data/121203.tar" "./data/OSSE/obs"

process_tar "https://www.seanoe.org/data/00966/107806/data/121238.tar" "./data/OSSE/ref"

process_tar "https://www.seanoe.org/data/00966/107806/data/127069.tar" "./data/OSSE/lowpass_ref_bm"

echo ""
echo "----------------------------------------"
echo "All downloads and extractions completed!"
echo "----------------------------------------"