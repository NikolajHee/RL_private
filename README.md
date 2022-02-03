# 02465 Introduction to reinforcement learning and control theory
This repository contains code for 02465, introduction to machine learning and control theory.

## Reading material
See DTU Learn for reading material, exercise instructions, and lectures handouts. 

# Installation guide
The installation involves two steps. First, you need to install the source code itself, and then you will possibly miss some external dependencies required to use certain features of openai Gym.
Complete the installation of the toolbox first, then install the dependencies as needed during the course based on your OS (Windows/Linux/OS-X).

## Course toolbox installation
The software toolbox for the course can be found in the `irlc` subdirectory and is distributed as a python package. 

 - **For details on using the software, please see the two videos on DTU Learn under 'Videos'. Please do this, and install the software, before the first lecture/exercise session.**
 - Exercise 0 on DTU learn contains additional information and a self-test you should try before the first lecture

### Unitgrade
In order to create the assignments you need unitgrade: https://lab.compute.dtu.dk/tuhe/unitgrade
Full installation instructions are provided through DTU Learn, but it should be as simple as:
```
pip install unitgrade   
```
In case there is a new version you can upgrade using:
```
pip install unitgrade --upgrade --no-cache
```

# External dependencies
I will provide instructions on how to do this in windows, since Linux and OS-X is much simpler and the error messages from Gym will explain what you should do (furthermore I don't have OS-X).
Please don't hesitate to post on e.g. Piazza if you encounter problems or find solutions. 

# M1/ARM chipset (Mac os-x)
Apple recently upraded from intel to an ARM chipset. This means a lot of disruption and the software ecosystem gets up to date. I am certain the toolbox can run under os-x since so many developers in machine learning uses mac, however, it will mean some additional setup is required. 
If you already have python 3.8 or later working and able to install packages it should be fine, otherwise you have to set up python 3.8 and install the missing packages. Links to get you started:

To setup python on M1:
- https://towardsdatascience.com/how-to-easily-set-up-python-on-any-m1-mac-5ea885b73fab

To install packages form pythons main repository pypi.org on anaconda
 - https://stackoverflow.com/questions/29286624/how-to-install-pypi-packages-using-anaconda-conda-command

Select the anaconda interpreter in pycharm. See:
 - https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html

## ffmpeg (Windows)
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
(see next topic)
### Missing Swig (Windows)
On windows, gym-box2d requires `swig.exe`, and you will get a long error that includes `error: command 'swig.exe' failed:` when running the above command. To fix this:
 - Download Swig 3.0.12 from: https://sourceforge.net/projects/swig/files/swigwin/swigwin-3.0.12/ (Note this is not the newest version, but swig 4.x.x does not appear to work)
 - Unzip and put the folder somewhere sensible, for instance the root of your drive. You should now have `swig.exe` at for instance:  `C:/swigwin-3.0.12/swig.exe`
 - Add the folder containing `swig.exe` to your `PATH` using the same method as for ffmpeg (se link to guide). When you open a **new** terminal, you can now run `swig.exe`. 
 - Re-run `pip install gym[box2d]` 


## cvxpy (Windows)
Først skal man installerede nogen build tools fra Visual Studio fra linket her: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16
Under installationen skal man vælge hvilke pakker man vil bruge. Vi vil gerne have dem markeret her:
https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view
Når de er installeret skal man til sidst i pycharm vinduet skrive
```
pip install cvxpy 
```


