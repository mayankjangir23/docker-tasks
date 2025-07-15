# Use a base image with GUI capabilities
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Firefox and GUI dependencies
RUN apt update && apt install -y \
    firefox \
    dbus-x11 \
    x11-utils \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libatk1.0-0 \
    libnss3 \
    libxss1 \
    libxtst6 \
    libx11-xcb1 \
    fonts-dejavu \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Default command
CMD ["firefox"]
