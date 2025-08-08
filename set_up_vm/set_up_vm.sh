#!/bin/bash

# Load .env file (assuming it's in the same directory)
set -a
source ../.env
set +a

# SSH into the VM and run setup commands
ssh -i $PRIVATE_KEY_PATH $USERNAME@$VM_IP_5 << EOF
#  # Install git
#  sudo apt update
#  sudo apt install -y git python3-venv
#
##  # Clone repo with authentication
#  git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/KirillUtyashev/orchard-action-market.git
#
#  # Setup virtual environment
  cd orchard-action-market
#  git pull
#  python3 -m venv venv
  source venv/bin/activate

  # Install requirements
  pip install --upgrade pip
  pip install -r requirements.txt

  echo "Setup complete."
EOF
