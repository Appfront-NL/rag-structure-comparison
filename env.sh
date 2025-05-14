#!/bin/bash

ENV_NAME="myenv"
YAML_FILE="environment.yml"

install() {
    if conda env list | grep -qE "^$ENV_NAME\s"; then
        echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
    else
        echo "Creating Conda environment from $YAML_FILE..."
        conda env create -f "$YAML_FILE" -n "$ENV_NAME"
    fi
}

run() {
    echo "Running script using Conda environment '$ENV_NAME'..."
    conda run -n "$ENV_NAME" python comparison/euTasksList.py
}

clean() {
    echo "Removing Conda environment '$ENV_NAME'..."
    conda remove -y -n "$ENV_NAME" --all
}

# Dispatcher
case "$1" in
    install)
        install
        ;;
    run)
        run
        ;;
    clean)
        clean
        ;;
    *)
        echo "Usage: $0 {install|run|clean}"
        exit 1
        ;;
esac
