# adaptive-hexapod-control
adaptive control and simulation for hexapod mk-III

## Model
Stl files was extracted from the complete hexapod model using fusion 360 and broght together using mojuco's xml.\
The body was seperated from the legs and each leg was separated to 3 pieces according the joints.\
<img src="images/legstructure.jpg" width=200>
### Hexapod mk-III model
<img src="images/hexapod.gif" width=200>

### Body stl
<img src="images/body.gif" width=200>


### Coxa stl
<img src="images/coxa.gif" width=200>


### Femur stl
<img src="images/femur.gif" width=200>


### Tibia stl
<img src="images/tibia.gif" width=200>

### Gaits
currently, 3 basic walking gait was implements:
<img src="images/gaits.png" width=200>
<img src="images/walking_gaits.gif">

## Loop
The simulation is based on the following loop as seen in main.py:<br/>
1. load mujoco model from xml.
2. get state
3. generate action (joints target angles)
4. step hexpod into this direction and calculate joint's speed.
5. goto 2.

## Structure
directory structure is as follows:<br/>
* gait: motion script and action generator.
* kinematics: inverse and forward kinematics for leg swing use (input: joint angles, target position. output: joints angles to get to target).
* model: model of hexapod: environment, legs and joints description.
* mujoco models: xml and stl files to  load physical model.
* main: main file, starts simulation.
