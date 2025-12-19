#!/bin/bash
# Setup script for Semantic Correspondence baseline

echo "Setting up Semantic Correspondence environment..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download SPair-71k dataset to data/SPair-71k/"
echo "3. Run evaluation: python eval.py"
echo ""
echo "For testing without dataset: python eval.py --data_dir data/dummy --no-wandb"
