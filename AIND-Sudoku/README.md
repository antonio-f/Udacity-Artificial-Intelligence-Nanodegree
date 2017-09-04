# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Constraint propagation is a type of inference in which the constraints are used to reduce the number of legal values for a variable, which in turn can reduce the legal values
for another variable, and so on. 
A constraint is a relation defined on a sequence of variables (x_1, x_2, ..., x_t) where t is a suitable natural number. So a constraint is the subset of the powerset N^t formed by all t-tuples satisfying the relation. In practice this set-theoretical definition is not always useful, but we can have a clear view about the constraints from the clauses that define them. For example, in Sudoku we have 81 variables with domain {1,...9} and initially (non-diagonal case) not-equals constraints on the rows, columns, and 3x3 boxes, e.g.

alldifferent(( X_11 , X_21 , X_31 , ..., X_91 ),... etc.);
alldifferent(( X_11 , X_12 , X_13 , ..., X_19 ),... etc.);
alldifferent(( X_11 , X_21 , X_31 , X_12 , X_22 , X_32 , X_13 , X_23 , X_33 ),... etc.).

For naked twins problem, constraint propagation works by reducing domains of variables at each transformation: if a unit includes two boxes containing two identical digits, the remaining boxes in that unit must not contain these digits. This constraint propagates at every puzzle reduction step to all units (i.e. all rows, columns, 3x3 squares, and - in a diagonal Sudoku - the two diagonals). 

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A:  We add a new type of unit, that is the diagonal unit. For the diagonal Sudoku problem we use the same constraint propagation method as for regular Sudokus but applied to the 2 diagonals too.


### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the `assign_value` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login) for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

