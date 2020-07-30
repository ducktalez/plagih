#!/usr/bin/env bash
#-*- coding:utf-8 -*-
printf "Starting run $1 at $PWD/$1"
python3 start.py -config_lookup $1 $2 $3 $4