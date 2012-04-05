#!/bin/bash
rm memory_report.txt
touch memory_report.txt
for (( ; ; ))
do
   ps -eo pid,pri,pcpu,size,cmd --sort pcpu | tail >> memory_report.txt
   sleep 1m
done
