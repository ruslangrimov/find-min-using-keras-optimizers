# find-min-using-keras-optimizers
Finding the minimum of an arbitrary function using optimizers from Keras

### Overview

This python notebook shows how to use optimizers from Keras to find the minimum of an arbitrary function.

Here is a piece of code demonstrating using of SGD optimizer to take several steps towards the minimum of the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) from start point:
```python
# try an optimizer
opt = optimizers.SGD(nesterov=True, momentum=0.9, lr=0.001)
steps = find_min(px, py, opt)
visualize_path(steps)
print_path(steps)
```
![Plot](https://github.com/ruslangrimov/find-min-using-keras-optimizers/blob/master/path_plot.png?raw=true)
```
f: 692.875000,	x: -2.500000,	y: -2.000000
f: 67.764633,	x: -0.919200,	y: -1.686500
f: 22.453543,	x: -0.061143,	y: -1.456656
f: 27.118574,	x: 0.634704,	y: -1.239867
f: 57.453159,	x: 1.177989,	y: -1.008621
f: 87.557968,	x: 1.488615,	y: -0.739009
f: 77.580658,	x: 1.533951,	y: -0.427203
f: 43.270092,	x: 1.407844,	y: -0.094119
f: 16.090534,	x: 1.225150,	y: 0.234507
f: 3.105619,	x: 1.047903,	y: 0.541026
f: 0.010240,	x: 0.900097,	y: 0.815265
f: 1.889213,	x: 0.788901,	y: 1.051859
f: 5.517203,	x: 0.715033,	y: 1.248565
f: 9.068881,	x: 0.677123,	y: 1.405313
f: 11.558311,	x: 0.673464,	y: 1.523680
f: 12.477252,	x: 0.702523,	y: 1.606587
f: 11.643894,	x: 0.762702,	y: 1.658171
f: 9.213374,	x: 0.851476,	y: 1.683725
f: 5.783146,	x: 0.963994,	y: 1.689669
f: 2.430155,	x: 1.091450,	y: 1.683381
```


