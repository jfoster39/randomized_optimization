#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=/Users/tylerfoster/.ABAGAIL/ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

# four peaks
echo "four peaks"
echo
jython fourpeaks.py

# count ones
echo "count ones"
echo
jython countones.py

# continuous peaks
echo "continuous peaks"
echo
jython continuouspeaks.py

# knapsack
echo "Running knapsack"
echo
jython knapsack.py

# Traveling Salesman
echo "Running traveling salesman" 
echo 
jython travelingsalesman.py 

