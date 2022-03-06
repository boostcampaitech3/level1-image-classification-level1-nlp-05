#!/bin/bash

while  read line
do 
    python inference.py $line

done < command_inference.txt