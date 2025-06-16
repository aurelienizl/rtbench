#!/bin/bash

set -e

REQUIRED_PACKAGES=(
    python3
    python3-venv
    build-essential
    libnuma-dev
    stress-ng
)

install_packages() {
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! dpkg -l | grep -qw "$package"; then
            echo "Installing $package..."
            sudo apt-get install -y "$package"
        else
            echo "$package is already installed."
        fi
    done
}

create_virtualenv() {
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    else
        echo "Virtual environment already exists."
    fi
}

activate_virtualenv() {
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source venv/bin/activate
}

install_python_packages() {
    echo "Installing Python packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

main() {
    install_packages
    create_virtualenv
    activate_virtualenv
    install_python_packages
    echo "Setup complete!"
}

main "$@"