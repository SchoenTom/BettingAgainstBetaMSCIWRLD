#!/bin/bash
# Install script that bypasses pip cache for fresh package downloads

echo "Installing dependencies without cache..."
pip install --no-cache-dir -r requirements.txt

echo "Installation complete!"
