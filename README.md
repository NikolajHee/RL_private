# 02465 Introduction to reinforcement learning and control theory

This repository contains code for 02465, introduction to machine learning and control theory.

## Reading material
See DTU Learn for reading material, exercise instructions, and lectures handouts. 

# Installation guide
The installation involves two steps. First, you need to install the source code itself, and then you will possibly miss external dependencies required to use certain features of openai Gym.
Complete the installation of the toolbox first, then install the dependencies as needed and based on your OS (Windows/Linux/OS-X).

## Course toolbox installation
The software toolbox for the course can be found in the `irlc` subdirectory and is distributed as a python package. 

 - **For details on using the software, please see the two videos on DTU Learn under 'Videos'. Please do this, and install the software, before the first lecture/exercise session.**

If you are stuck with an exercise, please do not hesitate to contact me for a solution. Likewise if you would like the code for one of the in-class demos. 

### Unitgrade
In order to create the assignments you need unitgrade: https://lab.compute.dtu.dk/tuhe/unitgrade
Full installation instructions are provided in the second of the abovementioned videos on DTU Learn or the online installattion guide but it should be as simple as:
```
pip install git+https://git@gitlab.compute.dtu.dk/tuhe/unitgrade.git   
pip install git+https://git@gitlab.compute.dtu.dk/tuhe/unitgrade.git --upgrade 
```
The later option is for upgrading.

## External dependencies
I will provide instructions on how to do this in windows, since Linux and OS-X is much simpler and the error messages from Gym will explain what you should do (furthermore I don't have OS-X).

### ffmpeg (Windows)
- Go to https://ffmpeg.org/download.html#build-windows and select a source. In my case I went with  https://www.gyan.dev/ffmpeg/builds/ and selected the first link (https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z).
- Unzip the folder. For simplicity, rename it to `ffmpeg` and put it somewhere. You should now have a folder called something like
```
C:\Users\tuhe\Documents\2021\ffmpeg\bin
```
with the ffmpeg executable. 
- you have to add this location to your path. There are a multitude of guides for doing this but I followed: https://stackoverflow.com/questions/44272416/how-to-add-a-folder-to-path-environment-variable-in-windows-10-with-screensho
- You are done! open a new command prompt and type in `ffmpeg` and check your installation works (may require restart)

## Lunar Lander
The lunar-lander problem (in the control section) requires that you install the gym-box2d environments. To do so, run
```
pip install gym[box2d]
```
(see below)
### Missing Swig (Windows)
On windows, gym-box2d requires `swig.exe`, and you will get a long error that includes `error: command 'swig.exe' failed:` when running the above command. To fix this:
 - Download Swig 3.0.12 from: https://sourceforge.net/projects/swig/files/swigwin/swigwin-3.0.12/ (Note this is not the newest version, but swig 4.x.x does not appear to work)
 - Unzip and put the folder somewhere sensible, for instance the root of your drive. You should now have `swig.exe` at for instance:  `C:/swigwin-3.0.12/swig.exe`
 - Add the folder containing `swig.exe` to your `PATH` using the same method as for ffmpeg (se link to guide). When you open a **new** terminal, you can now run `swig.exe`. 
 - Re-run `pip install gym[box2d]` 


# cvxpy (Windows)
Først skal man installerede nogen build tools fra Visual Studio fra linket her: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16
Under installationen skal man vælge hvilke pakker man vil bruge. Vi vil gerne have dem markeret her:
https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view
Når de er installeret skal man til sidst i pycharm vinduet skrive
```
pip install cvxpy 
```

