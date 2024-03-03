#!/bin/bash

# Predefined download link
download_link="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"

# Extract the file name from the download link
file_name=$(basename "$download_link")

# Download the file
echo "Downloading $file_name..."
wget "$download_link"

# Extract the file
echo "Extracting $file_name..."
tar -xvzf "$file_name"

# Check if the extraction was successful
if [ $? -ne 0 ]; then
    echo "Error extracting the file."
    exit 1
fi

# Delete the downloaded file
echo "Deleting $file_name..."
rm "$file_name"

echo "Process completed successfully."
