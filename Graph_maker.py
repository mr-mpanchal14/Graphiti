from numpy import *
import tkinter as tk
from functools import partial
import matplotlib.pyplot as plt
import re

def call_result(eq, n1, n2, d):
    eq = (eq.get()) # get equation as string
    num1 = int((n1.get())) # get lower limit
    num2 = int((n2.get())) # get upper limit
    div = int((d.get())) # get divisions
    l = linspace(num1, num2, div) # convert list from limit and divisions
    ans = []
    # variables for dynamic display
    xpos = False
    xneg = False
    pos = False
    neg = False
    for i in l:
        if i>0: # if negative x co-ordinate is present
            xpos = True
        elif i<0:
            xneg = True
        temp = re.sub('k', f"{i}", eq)
        ans.append(eval(temp))
        if ans[-1] >=0: # if negative y co-ordinate is presetn
            pos = True
        else:
            neg = True

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # make a subplot
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    if not xneg or not xpos: #general view
        ax.spines['left'].set_position('zero')
    else: # 4 quadrant view
        ax.spines['left'].set_position('center')
    if not neg: # if y is fully non-negative
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    elif not pos: # if y in fuully non-positive
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')

    plt.plot(l, ans, 'g')
    plt.show()



root = tk.Tk() #intilaise TK var
root.geometry('400x250+100+200') # define size of window

root.title('Calculator') #defien title

eqn = tk.StringVar() #define variable for equation
num1 = tk.StringVar() # define variable for upper limit
num2 = tk.StringVar() # define variable for lower limit
div = tk.StringVar() #define variable for number of divisions

# defining Labels for variables
leqn = tk.Label(root, text="Enter the Equation in Terms of Variable k").grid(row=1, column=0, pady=5)
lNum1 = tk.Label(root, text="Enter the Starting Limit").grid(row=3, column=0)
lNum2 = tk.Label(root, text="Enter the Ending Limit").grid(row=4, column=0)
ldiv = tk.Label(root, text="Enter the Number of Divisions").grid(row=5, column=0)


labelResult = tk.Label(root) #define labels for result

#define entry with positioning
eeqn = tk.Entry(root, textvariable=eqn).grid(row=2, column=0, padx=10, ipadx=50, pady=15)
eNum1 = tk.Entry(root, textvariable=num1).grid(row=3, column=2, pady=5)
eNum2 = tk.Entry(root, textvariable=num2).grid(row=4, column=2, pady=5)
ediv = tk.Entry(root, textvariable=div).grid(row=5, column=2, pady=5)

call_result = partial(call_result, eqn, num1, num2, div) # Call Results graph

buttonCal = tk.Button(root, text="Graffiphy", command=call_result).grid(row=9, column=0, pady=15) # call button

root.mainloop() # call tkinter window