# Point Cloud Regional Matching with Simulated Annealing

This code base aims to improve the KNN point cloud labeling by matching regional features using Simulated Annealing.

## Data

This project uses the [ShapeNet Part dataset](https://cs.stanford.edu/~ericyi/project_page/part_annotation/). Part of the data processing uses the script found in this repository:

https://github.com/Donglin-Wang2/shapenet-partnet-utils

The rest of the preprocessing is done by the `preprocessing.py` script in this repository.

If you wish to download the processed data, please [click here](https://drive.google.com/file/d/1wrsP83sUb1vE-38Mr29f4dBIhehO1S8_/view?usp=sharing). After downloading, please unzip the content to the `./data` folder.

## Dependencies

If you wish the install all dependencies in your local environment using conda, please type the commands in `Dockerfile` without the `RUN` and `FROM` keywords. Otherwise, the following is an instruction on creating a Docker container for dependencies. Note that this has not been tested. In addition, you will have to move the content of this repository into the container after the image has been built.

1. Install Docker.
2. Go to the root folder of this repo.
3. Run the following command. You have to replace `<tag_name>` with a name of your choice. Once done, you will see a new image show up in your Docker desktop portal. 
```
docker build -t <tag_name> .
```
4. Run the docker image using the following command, where `<tag_name>` is the name you chose in the previous step.
```
docker run -i -t <tag_name> /bin/bash
```