FROM continuumio/miniconda3

RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -c bottler nvidiacub
RUN conda install pytorch3d -c pytorch3d
RUN conda install -c conda-forge trimesh
RUN pip install open3d