#!/usr/bin/env bash
#-*- coding:utf-8 -*-
printf "Starting run $1 at $PWD/$1 with options: -less_files"
python3 plagih_gp.py -config_lookup -less_files $1 $2 $3 $4