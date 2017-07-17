#!/bin/bash

python make_files_nb.py

atom=TIP4P2005
at=tip4p2005

for c in {8..8}
do
	for i in {0..0}
	do
		let i=i*5+150
		chmod +x ../NANOBUBBLES/${atom}/CUT_${c}_A/${atom}_${i}/EQ1/eq1_${at}_${i}.sh
		chmod +x ../NANOBUBBLES/${atom}/CUT_${c}_A/${atom}_${i}/EQ2/eq2_${at}_${i}.sh
		chmod +x ../NANOBUBBLES/${atom}/CUT_${c}_A/${atom}_${i}/BUBBLE/bubble_${at}_${i}.sh
	done
done
