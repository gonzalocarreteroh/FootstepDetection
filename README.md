# FootstepsDetection

A full description of our project can be found [here](docs/COMP4531Project.pdf)

In this project, we utilize the intrinsic unique characteristics of a person's footstep's acoustic
signals in order to identify different individuals in real time. Making use of acoustic signals
serves as a less intrusive way of identification and is robust to visual occlusions or low light
conditions compared to using cameras. The applications can span from integration into smart
home/building security systems to improving robots awareness for personalized interactions
as well as supplement other sensors for multiple object tracking under adverse visual
scenarios.

Our system records the noise generated in an enclosed environment and sends it to our server
for real time classification using a CNN model that extracts characteristic embeddings after data preprocessing. We have collected our own data samples in
order to perform identification of a reduced number of subjects.
