# Note: You can use any Debian/Ubuntu based image you want. 
FROM nvcr.io/nvidia/tensorrt:23.07-py3
# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends ffmpeg libsm6 libxext6