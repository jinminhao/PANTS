#!/usr/bin/env bash
eval "$(conda shell.bash hook)"

conda create -y -n py39-app-vpn python=3.9

conda activate py39-app-vpn

pip3 install -r requirements-app-vpn.txt

conda create -y -n py39-vca python=3.9

conda activate py39-vca

pip3 install -r requirements-vca.txt