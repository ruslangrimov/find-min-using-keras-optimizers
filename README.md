# find-min-using-keras-optimizers
Finding the minimum of an arbitrary function using optimizers from Keras

### Overview

This python notebook shows how to use optimizers from Keras to find the minimum of an arbitrary function.

Although it can be done using pure TensorFlow or Theano it is better to have high level code which enables you to switch between backends whenever you want. In addition, Theano unlike TensorFlow doesn't have its own builtin optimizers so you would have to implement it yourself.

### Using keras.optimizers

To use optimizers it is necessary first to import them and keras backend:
```python
from keras import backend as K
from keras import optimizers
```

The second step is the definition of a function that will be optimized. For example consider the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).

Define a native python function:
```python
a = 1
b = 10

def f(x, y):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2
```

Define parameters of that function with initial values from which the optimizer will start its path to function minimum:
```python
X = K.variable(-2.5, name='x')
Y = K.variable(-2.0, name='y')
```

Make a keras function. As keras.variable objects support standart math operators in our case we can just pass them to the ```f``` to obtain the keras function:
```python
F = f(X, Y)
```

If your function contains some non-standard operations you may call necessary methods from ```backend``` package. See the [documentation](https://keras.io/backend/#backend-functions) for more info.

Unlike a native python function ```F``` isn't intended to be called. This is a special object &mdash; tensor graph.

The third step is creation of an optimizer object and functions which will be used to update parameters.

Initialize chosen optimizer object with parameters:
```python
opt = optimizers.SGD(nesterov=True, momentum=0.9, lr=0.001)
```

Make an array of functions which updates the parameters ```X``` and ```Y``` of ```F``` to minimize ```F```'s value using given optimizer:
```python
updates = opt.get_updates([X, Y], [], F)
```

Make a function that returns the ```F```'s value and updates the parameters:
```python
iterate = K.function([], [F], updates)
```
You may add any parameters you want to watch over during the optimization in the second parameter of ```K.function```.

The last step is optimization. Call ```iterate``` several times or until F_value achieves the desired minimum:
```python
for i in range(20):
    F_value = iterate([])[0]
    if F_value < 0.01:
        break
```

Here is a piece of code demonstrating using of SGD optimizer to take several steps towards the minimum of the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) from start point (see realization of find_min and others functions in the notebook above):
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

### Optimizing a function with a vector of parameters
There is used a more involved variant of the Rosenbrock function with many parameters. See the formula in the notebook.


```python
a = 1
b = 100

pn = 1024  # Number of elements in the vector

# initialize a vector with random numbers
X = K.variable(np.random.uniform(-5, 5, size=pn), name='x')

# define a function
F = K.sum(b * (X[1::2] - X[::2] ** 2) ** 2 + (a - X[::2]) ** 2)

# initialize an optimizer
opt = optimizers.Adagrad(lr=2.5)

# make a function that updates parameters and return the function value
updates = opt.get_updates([X], [], F)
iterate = K.function([], [F], updates)

# run gradien descent
for i in range(50):
    F_value = iterate([])[0]
    print('step %2d:\tvalue: %f' % (i, F_value))
```

```
step  0:	value: 7068827.500000
step  1:	value: 1497900.750000
step  2:	value: 325722.625000
...
step 47:	value: 1343.550171
step 48:	value: 1342.815918
step 49:	value: 1342.091064
```
