#!/bin/bash

DET_MASTER="http://localhost:4363"
DET_USER="bschilling"

case "$1" in
	r|run)
		shift
		PYTHONPATH="$PWD/src" python3 src/scripts/main.py "$@"
		;;
	td)
		shift;
		PYTHONPATH="$PWD/src" python3 src/scripts/transform_dataset.py "$@"
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
		PYTHONPATH="$PWD/src" python3 src/scripts/inspect_model.py "$@"
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
		# det -m "$DET_MASTER" -u "$DET_USER" experiment create ./configs/imagenet_asha.yaml ./src
		det -m "$DET_MASTER" -u "$DET_USER" experiment create ./configs/imagenet_single.yaml ./src
		;;
	docker)
		docker run --rm -it -v /home/alok/data:/data -v "$PWD":/workspace/mltestcase bruno1996/determined-pytorch:0.1 bash
		;;
	j|jupyter)
		jupyter lab
		;;
	*)
		echo "choose between: train, eval, inspect, vis, playground, test, d, jupyter"
		;;
esac
