# Body Leveling key points

Given the hexapod in a certain position.
We monitor the height (and constant width) in each connection to the body i.e. coxa.

Let's first consider a simple scenario in 2D:
<img src="../images/body_leveling.jpg" width=200>

The axes in which we consider the problem are y and z (x is equal) so: <img src="https://render.githubusercontent.com/render/math?math=h=h_2-h_1"> and <img src="https://render.githubusercontent.com/render/math?math=w=w_2-w_1"> but this should be constant. Thus, the angle of rotation is <img src="https://render.githubusercontent.com/render/math?math=\theta=arctan(\frac{h}{w})">.

The algorithm for body leveling given <img src="https://render.githubusercontent.com/render/math?math=\theta"> is:
1. Get <img src="https://render.githubusercontent.com/render/math?math=h_1,w_1,h_2,w_2"> (by sensing with SNN or direct simulation output)
2. calculate <img src="https://render.githubusercontent.com/render/math?math=\theta">
3. get target orientation <img src="https://render.githubusercontent.com/render/math?math=\theta'"> (in out case is hard-coded 0 but it can be an output of SNN as well)
4. for each leg i:
   - get current angles of joints  <img src="https://render.githubusercontent.com/render/math?math=q_i">
   - using forward kinematics calculate the radius in y,z plane using <img src="https://render.githubusercontent.com/render/math?math=q_i">
   - if the leg is in the left side (yz plane spesific) <img src="https://render.githubusercontent.com/render/math?math=\theta=\theta-\pi'"> and  <img src="https://render.githubusercontent.com/render/math?math=\theta'=\theta'-\pi'">
   - calculate cartesian change of the leg given a rotation around the body center with the radius and <img src="https://render.githubusercontent.com/render/math?math=\theta'"> (convert polar to cartesian and return the difference)
   - extend the 2d to 3d by setting <img src="https://render.githubusercontent.com/render/math?math=x=0">, the result is difference 3d vector <img src="https://render.githubusercontent.com/render/math?math=d_i">
   - using inverse kinematics, calculate the angles which will make the leg go from it's current position to <img src="https://render.githubusercontent.com/render/math?math=-d_i">
   - set new angles from inverse kinematics

## Results

<img src="../images/body_level_impl_before.png" width=200><img src="../images/body_level_impl_after.png" width=200>

