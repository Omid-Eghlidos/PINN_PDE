# PINN for Solving PDE

**P**hysics-**I**nformed **N**eural **N**etwork (PINN) for Solving 
**P**artial **D**ifferential **E**quations (PDE) Problem

This repository contains a program for developing PINN for solving a few of the famous PDEs including 
Lid-Driven Cavity (LDC).

## Requirements

1) Python3 with latest version of following libraries (as of Feb. 2024):

* Numerical Python (numpy)
* Scientific Python (scipy)
* Python Torch (pytorch)
* Mathematical Plotting Libraray (matplotlib)

2) Grpahic Card (NVIDIA) (optional)

## Inputs

An input script with the desired settings as provided in the template.

## Deployment

In the deignated folder run: [path]/run.py -t [type] -i [input] -v

* path -> Path to this code on your system
* -i input -> Input file with desired settings as provide in the template
* -t type -> Type of architecture to be used: currently only "pinn"
* -v -> Activate verbose mode, which output more details on the screen

