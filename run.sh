#!/bin/bash

case "$1" in
	t|train)
		python3 src/train.py
		pling
		;;
	e|eval)
		python3 src/eval.py
		;;
	vis)
		shift
		python3 src/run_visualization.py "$@"
		;;
	*)
		echo "choose train or eval"
		;;
esac
