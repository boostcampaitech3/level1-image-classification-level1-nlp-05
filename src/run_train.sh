#!/bin/bash

while  read line
do 
    python train.py $line

done < command_file.txt