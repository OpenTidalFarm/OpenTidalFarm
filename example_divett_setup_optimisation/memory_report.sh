#!/bin/bash
rm -f memory_report.txt
touch memory_report.txt
for (( ; ; ))
do
   date >> memory_report.txt 
   ps -eo pid,pri,pcpu,size,cmd --sort pcpu | tail >> memory_report.txt
   sleep 1m
done
