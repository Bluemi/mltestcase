#!/bin/bash

case "$1" in
	t|train)
		shift;
		python3 src/train.py "$@"
		pling
		;;
	e|eval)
		shift;
		python3 src/eval.py "$@"
		;;
	vis)
		shift
		python3 src/run_visualization.py "$@"
		;;
	te|test)
		shift;
		python3 src/test.py "$@"
		pling
		;;
	d)
		det -m "$DET_MASTER" -u "$DET_USER" experiment create ./configs/default.yaml ./src
		;;
	*)
		echo "choose train or eval"
		;;
esac
