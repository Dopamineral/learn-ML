# Chapter 2: Training Simple Machine Learning Algorithms for Classification

## Exercise 1 : Make a virtual environment for your ML projects

- make a virtual environment in python
- activate it
- make a requirements file with numpy, pandas, matplotlib
- install the requirements file


??? abstract "Solution"

    === "hint"
        google python venv


    === "Solution"

        ```sh
        python3 -m venv .venv
        source .venv bin activatre
        touch requirements.txt

        # insert numpy into requirements.txt
        # insert pandas into requirements.txt
        # insert matplotlib into requirements.txt

        ```
        ```sh
        pip install -r requirements.txt

        ```

## Exercise 2: Warm up
- make a main.py file
- add function that iteratively adds 1 to a number of n amount of cycles. e.g "my number is a, for something in range n: add 1 to my number.
- add 'verbose' as key word argument to function that optionally prints out "hello" when the function is run


??? abstract "Solution"

    === "hint"
        google defining functions in python
        it helps if the verbose argument is a boolean


    === "Solution"

        ```sh
        # make the file
        touch main.py
        ```

        ```py title="main.py"

        def add_stuff(a,n, verbose=False):
            if verbose:
                print("hello")

            for ii in range(n):
                a += 1
                print(a)

            return a

        ```

## Exercise 3: Key concept 1: classes
- copy the python code below and try to complete what is missing
```py title="exercise 3"
class Neuron:
    def __init__(self, a):
        self.a_value = a

    def do_something(self):
        print(self.a_value)

    def calculate_something(self,a,b):
        print(a+b)

neuron1 = Neuron(2)
neuron1.do_something()
neuron1.calculate_something(2,3)


# Ex 1: Run the above stuff and try to explain what is going on

# Ex 2: Change the Neuron class so it takes in two arguments

# Ex 3: print out the internal values of your neuron object

# Ex 4: add a "calculate" function to the neuron class that multiplies its internal values
```


??? abstract "Solution"

    === "hint"
        cheating now are we?


    === "Solution"

        ```py

        # Ex 2: Change the Neuron class so it takes in two arguments
        class Neuron2:
            def __init__(self, a, b):
                self.a_value = a
                self.b_value = b

            def do_something(self):
                print(self.a_value)

            def calculate_something(self,a,b):
                print(a+b)

        neuron2 = Neuron2(2,3)
        neuron2.do_something()
        neuron2.calculate_something(2,3)

        # Ex 3: print out the internal values of your neuron object
        print(neuron2.a_value)
        print(neuron2.b_value)

        # Ex 4: add a "calculate" function to the neuron class that multiplies its internal values
        class Neuron3:
            def __init__(self, a, b):
                self.a_value = a
                self.b_value = b

            def do_something(self):
                print(self.a_value)

            def calculate_something(self,a,b):
                print(a+b)

            def calculate(self):
                product = self.a_value * self.b_value
                print(product)

        ```

## Exercise 4: funny thing: the zip function
- find out what the zip function does and add that to a neuron class to do something fancy

??? abstract "Solution"

    === "hint"
        [Realpython has a nice overview of what the zip function does](https://realpython.com/python-zip-function/#understanding-the-python-zip-function)


    === "Solution"

        ```py
        X = [0,1,2,3,4,5,6,7,8,9] # train data
        y = [0,1,0,1,0,1,0,1,0,1] # test data


        def combine_train_test(X,y):
            iterator = zip(X,y)
            list_of_iterator = list(iterator)
            print(list_of_iterator)

        ```

## Exercise 5: Copy paste the code from the handbook and correct the indentation.
- this is a nice exercise, for real

```py
import numpy as np
class Perceptron:
"""Perceptron classifier.
Parameters
------------
eta : float
Learning rate (between 0.0 and 1.0)
n_iter : int
Passes over the training dataset.
random_state : int
Random number generator seed for random weight
initialization.
Attributes
-----------
w_ : 1d-array
Weights after fitting.
b_ : Scalar
Bias unit after fitting.
errors_ : list
Number of misclassifications (updates) in each epoch.
"""
def __init__(self, eta=0.01, n_iter=50, random_state=1):
self.eta = eta
self.n_iter = n_iter
self.random_state = random_state
def fit(self, X, y):
"""Fit training data.
Parameters
----------
X : {array-like}, shape = [n_examples, n_features]
Training vectors, where n_examples is the number of
examples and n_features is the number of features.
y : array-like, shape = [n_examples]
Target values.
Returns
-------
self : object
"""
rgen = np.random.RandomState(self.random_state)
self.w_ = rgen.normal(loc=0.0, scale=0.01,
size=X.shape[1])
self.b_ = np.float_(0.)
self.errors_ = []
for _ in range(self.n_iter):
errors = 0
for xi, target in zip(X, y):
update = self.eta * (target - self.predict(xi))
self.w_ += update * xi
self.b_ += update
errors += int(update != 0.0)
self.errors_.append(errors)
return self
def net_input(self, X):
"""Calculate net input"""
return np.dot(X, self.w_) + self.b_
def predict(self, X):
"""Return class label after unit step"""
return np.where(self.net_input(X) >= 0.0, 1, 0)
```

## Exercise 6: plotting data
- use numpy to create a sin function dataset
- use matplotlib to plot that data
- add xlabel and y label
- add title to the plot
- (add a legend to the plot) -> extra challenge


??? abstract "Solution"

    === "hint"
        use numpy linspace to create the x-values
        then you can create y as y=sin(x)


    === "Solution"

        ```py
        import numpy as np
        import matplotlib.pyplot as plt

        #create the values
        x = np.linspace(1,10,100)
        y = np.sin(x)

        #plot the values
        plt.plot(x,y)

        # add labels
        plt.xlabel("my fancy x label")
        plt.ylabel("my fancy y label")

        # add title
        plt.title("my fancy title")

        # add legend
        plt.legend(['first thing that is plotted', "secodn thing that is plotted "])

        plt.show()

        ``` 

## Exercise 7: What does the numpy.where function do?
- make a function that uses this


??? abstract "Solution"

    === "hint"
        the function returns indexes


    === "Solution"
        ```py
        import numpy as np

        X = np.array([1,0,1,0,1,0,1,0,1,0])
        y = np.array([1,2,3,4,5,6,7,8,9,10])

        # check what this output does
        print(np.where(X*y==0))
        ```


