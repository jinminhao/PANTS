# PANTS

[[Arxiv](https://arxiv.org/abs/2409.04691)]

## Abstract

Multiple network management tasks, from resource
allocation to intrusion detection, rely on some form of
ML-based network traffic classification (MNC). Despite their
potential, MNCs are vulnerable to adversarial inputs, which
can lead to outages, poor decision-making, and security
violations, among other issues.

The goal of this paper is to help network operators assess
and enhance the robustness of their MNC against adversarial
inputs. The most critical step for this is generating inputs that
can fool the MNC while being realizable under various threat
models. Compared to other ML models, finding adversarial inputs against MNCs is more challenging due to the existence of
non-differentiable components e.g., traffic engineering and the
need to constrain inputs to preserve semantics and ensure reliability. These factors prevent the direct use of well-established
gradient-based methods developed in adversarial ML (AML).

To address these challenges, we introduce PANTS, a
practical white-box framework that uniquely integrates
AML techniques with Satisfiability Modulo Theories (SMT)
solvers to generate adversarial inputs for MNCs. We also
embed PANTS into an iterative adversarial training process
that enhances the robustness of MNCs against adversarial
inputs. PANTS is 70% and 2x more likely in median to
find adversarial inputs against target MNCs compared to
state-of-the-art baselines, namely Amoeba and BAP. PANTS
improves the robustness of the target MNCs by 52.7%
(even against attackers outside of what is considered during
robustification) without sacrificing their accuracy.

## Artifact Introduction

This artifact includes all the necessary
code and scripts for PANTS, along with supplementary resources such as datasets and well-trained models including
both vanilla and robustified ones. Additionally, it provides a
comprehensive roadmap for effectively utilizing PANTS.

## Artifact Setup

We will provide an instance in CloudLab which has the environment configured. For any evaluation which is not using the provided CloudLab instance, please follow the procedures to setup the environment.

> We recommend the testbed to have similar or identical performance as the instance d430 or c220g5 in CloudLab.

### Step 1:
Please download the asset [here](https://drive.google.com/file/d/1uD-CXqbdkl8voQkyXQxQ6KytRGpvFGm-/view?usp=sharing) and uncompress the `asset.tar.gz`. After uncompression, you should see an `asset` directory which includes a lot of data and models. 

### Step 2:

```
$ cd PANTS/

$ conda create -n py39-app-vpn python=3.9
$ pip3 install -r requirements-app-vpn.txt

$ conda create -n py39-vca python=3.9
$ pip3 install -r requirements-vca.txt
```

After setup, you can briefly test if the environment is correctly configured by running 

```
$ cd scripts/
$ bash test_env.sh
```

If the environment is installed successfully, you should see the output as follows
```
$ bash test_env.sh

...
Summary: ASR: 1.0, speed: 2.0500868564283423
...
Summary: ASR: 1.0, speed: 0.5714318749236006
```

## Artifact Evaluation

In order to have smooth artifact evaluation, we provide a script which automatically runs all the provided testing scripts to get the results
and generate the corresponding figures.

```
$ cd scripts/
$ bash test-all.sh
```

The script automatically generates multiple figures in `figures/`. Here is the breakdown of a mapping between the figures in the paper and the generated figures from the artifact.

| Figure in the paper    | Figure name in `figures/` |
| -------- | ------- |
| Fig. 5  | important_features_end-host.pdf,  important_features_in-host.pdf   |
| Fig. 6  | app_end_host_vanilla.pdf, app_in_path_vanilla.pdf, vpn_end_host_vanilla.pdf, vpn_in_path_vanilla.pdf, vca_end_host_vanilla.pdf, vca_in_path_vanilla.pdf    |
| Fig. 7  | adv_train.pdf    |
| Fig. 8  | robustfication_end-host.pdf    |
| Fig. 8  | robustfication_end-host.pdf    |
| Fig. 9  | netshare_mlp.pdf, netshare_rf.pdf, netshare_tf.pdf, netshare_cnn.pdf    |
| Fig. 10  | app_end_host_robustified.pdf, app_in_path_robustified.pdf, vpn_end_host_robustified.pdf, vpn_in_path_robustified.pdf, vca_end_host_robustified.pdf, vca_in_path_robustified.pdf   |
| Fig. 11  | threat.pdf   |
| Fig. 12 | NA. Please refer to `logs/{app,vpn,vca}/..._vanilla/results.txt` for generation speed.
| Fig. 13  | transferability.pdf   |