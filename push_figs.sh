#!/bin/bash

(
    cd $HOME/Documents/gh-papers/Experimentally-Tunable-QED-in-DO-QSI/ 
    git add FIG
    git pull
    git commit -a 
    git push
)
