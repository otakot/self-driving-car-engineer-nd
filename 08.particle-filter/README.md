# Kidnapped Vehicle Project

## Setup instructions

This project requires the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

Scripts install-ubuntu.sh and install-mac.sh can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems accordingly. For windows use either Docker, VMware, or [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/)


## Other Dependencies

* cmake >= 3.5
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
* gcc/g++ >= 5.4


## Build instructions

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make

Alternatively some scripts have been included to simplify this process, these can be leveraged by executing the following in the top directory of the project:

1. ./clean.sh
2. ./build.sh
3. ./run.sh


## Execution instructions

1. Start created particle filter executable from build folder:
  
   ./particle_filter
   
2. Start downloaded Term 2 Simulator application. Select requried resolution and press Play button.
   Switch to 'Project 3: Kidnaped Vehicle' using right arrow an press Select button.
   To run the filter press Start button. 
