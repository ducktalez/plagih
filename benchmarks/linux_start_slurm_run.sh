#!/usr/bin/env bash
#-*- coding:utf-8 -*-
printf "Starting run $1 at $PWD with options: $1 $2 $3 $4"
python3 plagih_gp.py -mp_cpu_cores_max 16 -config_lookup $1 $2 $3 $4
