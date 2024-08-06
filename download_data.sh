mkdir -p data
cd data

# PINF
wget https://rachelcmy.github.io/pinf_smoke/data/pinf_data.zip
unzip pinf_data.zip -d pinf
rm pinf_data.zip
wget -P pinf/Game https://raw.githubusercontent.com/RachelCmy/pinf_smoke/main/data/Game/info.json
wget -P pinf/ScalarReal https://raw.githubusercontent.com/RachelCmy/pinf_smoke/main/data/ScalarReal/info.json
wget -P pinf/Sphere https://raw.githubusercontent.com/RachelCmy/pinf_smoke/main/data/Sphere/info.json

# Neural Volumes
wget https://github.com/facebookresearch/neuralvolumes/releases/download/v0.1/experiments.tar.gz
tar -zxf experiments.tar.gz experiments/dryice1
mv experiments/dryice1 dryice1
rmdir experiments
rm experiments.tar.gz
