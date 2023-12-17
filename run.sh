#!/bin/bash

case "$1" in
	t|train)
		python3 src/train.py
		;;
	e|eval)
		python3 src/eval.py
		;;
	*)
		echo "choose train or eval"
		;;
esac
