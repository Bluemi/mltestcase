#!/bin/bash

DET_MASTER="http://localhost:4363"
DET_USER="bschilling"

case "$1" in
	r|run)
		shift
		python3 src/main.py "$@"
		;;
	t|train)
		shift;
		python3 src/train.py "$@"
		pling
		;;
	e|eval)
		shift;
		python3 src/eval.py "$@"
		;;
	im|inspect)
		shift;
		python3 src/inspect_model.py "$@"
		;;
	vis)
		shift
		python3 src/run_visualization.py "$@"
		;;
	p|playground)
		shift
		python3 src/run_playground.py "$@"
		;;
	te|test)
		shift;
		python3 src/test.py "$@"
		;;
	d)
		det -m "$DET_MASTER" -u "$DET_USER" experiment create ./configs/imagenet_single.yaml ./src
		;;
	j|jupyter)
		jupyter lab
		;;
	*)
		echo "choose between: train, eval, inspect, vis, playground, test, d, jupyter"
		;;
esac
