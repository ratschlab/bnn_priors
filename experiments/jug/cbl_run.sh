#!/usr/bin/env bash
set -euo pipefail

if [ -z "$1" ]; then
    echo "Usage: $0 jugfile.py"
    exit 1
fi

MY_PATH="$(dirname $0)"              # relative
MY_PATH="$( cd $MY_PATH && pwd )"  # absolutized and normalized
if [ -z "$MY_PATH" ] ; then
	# error; for some reason, the path is not accessible
	# to the script (e.g. permissions re-evaled after suid)
	echo "Could not find path of this file."
	exit 1  # fail
fi


for host in huygens cartwright laplace poisson vapnik gosset julia fields \
    ramanujan banach riemann euler robbins vartak curie sagarmatha ariadne \
    grothendieck babbage neumann gan bernoulli; do
    ssh "$host" 'tmux new-session -s 0 -d; top -n 1 -b | head -n 15; nvidia-smi'
    echo "For host $host, should I set CUDA_VISIBLE_DEVICES? ([0,1,2,...])? Should I skip?"
    read command
    if [ "$command" = "skip" ]; then
        echo "Skipping..."
    elif [ "$command" = "" ]; then
        ssh "$host" "tmux new-window -n $1; tmux send-keys \"workon py37\" ENTER \"jug execute $MY_PATH/$1\" ENTER"
    else
        ssh "$host" "tmux new-window -n $1; tmux send-keys \"export CUDA_VISIBLE_DEVICES=$command\" ENTER \"workon py37\" ENTER \"jug execute $MY_PATH/$1\" ENTER"
    fi
done
