# Introduction

This is an example implementation of a phase field model for cell motility on the GPU using CUDA. Feel free to use it as a starting point for your own projects, but **please** cite the paper it's based on:

` Kulawiak DA, Camley BA, Rappel W-J (2016) Modeling Contact Inhibition of Locomotion of Colliding Cells Migrating on Micropatterned Substrates. PLoS Comput Biol 12(12): e1005239. https://doi.org/10.1371/journal.pcbi.1005239`
 
Note, that this is NOT a full reimplementation and certain features are not implemented. Nor is it tested to the extent the original code was.

The shape of each cell is defined by a phase field $` \phi(x)`$ and their polarity is given by a polarity marker  $`\rho(x)`$. We also include a fluctuating inhibitor  $`I(x)`$ to describe noise in the system and to control the persistence of motion of a cell.


# Usage
Compile the program using the compile script (check if the architecture matches your card)  
Check the included example parameter file example_param.ini, move it to a folder of your choice and rename it to param.ini  
Start to program with ./name "folder"

You can provide start positions and directions with the files `AngleStartData.dat` and `CellPosStartData.dat` and flipping the respective options in the parameter file to `true`. Otherwise the programm will choose randomish start positions (with a not to clever algorithm).


# Todo
-   More documentation
-   Chemical interactions
-   Startscript
-   Scripts to analyse output
-   proper makefile
-   Unittests

