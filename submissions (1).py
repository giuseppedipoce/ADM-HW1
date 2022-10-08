#Say "Hello, World!" With Python
print("Hello, World!")

#Python If-Else
#!/bin/python3

import math
import os
import random
import re
import sys



if  __name__ == '__main__':
    n = int(input().strip())
if n%2!=0:
    print("Weird")
elif    n%2==0 and n in range(2,6):
    print("Not Weird")
elif  n%2==0 and n in range(6,21):
    print("Weird")
elif n%2==0 and n > 20: 
    print("Not Weird")
    



#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)
    

#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)

#Loops
if __name__ == '__main__':
    n = int(input())
for i in range(0,n):
    print(i**2)





#Write a function
def is_leap(year):
    leap=False
    if year%4==0 and year%100!=0:
        return True
    elif year%4==0 and year%400==0 and year%100==0:
        return True 
    else:
        return False
    


#Print Function
if __name__ == '__main__':
    n = int(input())
for i in range(0,n):
    print(i+1,end='')
    

#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    new_list=[[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) 
    if (i+j+k)!=n]
print(new_list)
    




#Find the Runner-Up Score!  
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    nuovo_arr=set(arr)
    nuovo_arr.remove(max(nuovo_arr))
    print(max(nuovo_arr))
    



#sWAP cASE
def swap_case(s):
    return s.swapcase()



#String Split and Join
def split_and_join(line):
    line=line.split(" ")
    line1="-".join(line)
    return line1
    

#Nested Lists
if __name__ == '__main__':
    mylist=list()
    for _ in range(int(input())):
        name = input() 
        score = float(input())
        mylist.append([name,score]) #list of [name-score]
    min_list = min(x[1] for x in mylist) #take a list of min scores
    final_list=[x for x in mylist if x[1]> min_list] 
    min_list= min(x[1] for x in final_list)
    final=sorted(x[0] for x in final_list if x[1]== min_list)
"print(final)"
for i in final:
    print(i)

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    marks=student_marks[query_name]
print(format(sum(marks)/3,'.2f'))

#Lists
if __name__ == '__main__':
    N = int(input())
a=[]
for i in range(N):
    function_name,*num=input().split(" ")
    
    if function_name=="insert":
        i=int(num[0])
        e=int(num[1])
        a.insert(i,e)
    elif function_name=="print":
        print(a)
    elif function_name=="remove":
        a.remove(int(num[0]))
    elif function_name=="append":
        a.append(int(num[0]))
    elif function_name=="sort":
        a.sort()
    elif function_name=="pop":
        a.pop()
    elif function_name=="reverse":
        a.reverse()
    

    

#What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")


#Tuples 
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

t=tuple(integer_list)
print(hash(t))
#but it works only on pypy 3



#Mutations
def mutate_string(string, position, character):
    return string[:position]+character+string[position+1:]





#Find a string
def count_substring(string, sub_string):
    count=0
    for i in range(len(string)-len(sub_string)+1):
        if string[i:i+len(sub_string)]==sub_string:
            count +=1
          
    return count




#String Validators
if __name__ == '__main__':
    s = input()
    list=[i.isalnum() for i in s]
    print(any(list))
    list=[i.isalpha() for i in s]
    print(any(list))
    list=[i.isdigit() for i in s]
    print(any(list))
    list=[i.islower() for i in s ]
    print(any(list))
    list=[i.isupper() for i in s]
    print(any(list))

#Text Alignment
#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)) 

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap


def wrap(string, max_width):
    return textwrap.fill(string,max_width)








#Designer Door Mat ###i've seen a Youtube video on this exercise###
# Enter your code here. Read input from STDIN. Print output to STDOUT
n,m=map(int,input().split())

s2= 'WELCOME'
s1 = '.|.'
for i in range (n//2):
    print((s1*((i*2)+1)).center(m ,'-'))
    
print(s2.center(m ,'-'))

for i in range (n//2-1,-1,-1):
    print((s1*((i*2)+1)).center(m ,'-'))


#Alphabet Rangoli
import string

def print_rangoli(s):
    
    characters= " abcdefghijklmnopqrstuvwxyz"
    
    for i in range(s,0,-1):
        c=characters[s:i:-1] + characters[i:s+1]
        c='-'.join(c)
        print(c.center((s*4)-3,'-'))
    
    for i in range(0,s-1):
        c=characters[s:i+2:-1]+ characters[i+2:s+1]
        c='-'.join(c)
        print(c.center((s*4)-3,'-'))
        
        


#Capitalize!


# Complete the solve function below.
def solve(s):
    name=s.split(" ")
    string=''
    for i in name:
        string=string+ i.capitalize()+ ' '
        print(i)
    return string

    

#Introduction to Sets
def average(array):
    # your code goes here
    lst=set(array)
    avg_=sum(lst)/len(lst)
    return(avg_)
 
    
    
    
    
    
    

#Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
m=int(input())
m_set=set(map(int,input().split()))

n=int(input())
n_set = set(map(int,input().split()))

m_diff_n= n_set.difference(m_set)
n_diff_m= m_set.difference(n_set)

results= list(m_diff_n)+ list(n_diff_m)
sorted_res= sorted(results)
print(*sorted_res,sep='\n')



#No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n,m=map(int,input().split())
array=list(map(int,input().split()))

A=set(map(int,input().split()))
B=set(map(int,input().split()))

happ=sum((i in A)-(i in B) for i in array)
print(happ)


#Set .add() 
# Enter your code here. Read input from STDIN. Print output to STDOUT
c_stamps=int(input())
set1=set()
for i in range(c_stamps):
    set1.add(input())
print(len(set1))



#Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
m=int(input())
for i in range(m):
    s1=list(input().split())
    if s1[0]== "pop" :
        s.pop()
    elif s1[0]== "remove":
        s.remove(int(s1[1]))
    elif s1[0]== "discard":
        s.discard(int(s1[1]))

sum=0

for i in s:
    sum=sum+i
    
print(sum)
        
               



#Set .union() Operation
# Enter your code here. Read input from STDIN. Print outpu
n=map(int,input())
n_roll=set(map(int,input().split()))
b= int(input())
b_roll=set(map(int,input().split()))

AnB=n_roll.union(b_roll)
count=0
for i in AnB:
    count= count+1
print(count)


#Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
n_roll=set(map(int,input().split()))
b=int(input()) 
b_roll=set(map(int,input().split()))
 

AnB= n_roll.intersection(b_roll)
count=0
for i in AnB:
    count=count+1
print(count)

#Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
n_roll=set(map(int,input().split()))
b=int(input())
b_roll=set(map(int,input().split()))

AdifB = n_roll.difference(b_roll)
count=0
for i in range(len(AdifB)):
    count=count+1
print(count) 


#Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
n_roll=set(map(int,input().split()))
b=int(input())
b_roll=set(map(int,input().split()))

AsdB=n_roll.symmetric_difference(b_roll)

print(len(AsdB))


#Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
n_A=int(input())
st_A=set(map(int,input().split()))
N_operation=int(input())


for i in range(N_operation):
    operation,nb=input().split()
    st_B=set(map(int,input().split()))
    if operation=="intersection_update":
        st_A.intersection_update(st_B)
    elif operation=="update":
        st_A.update(st_B)    
    elif operation=="symmetric_difference_update":
        st_A.symmetric_difference_update(st_B)
    elif operation=="difference_update":
        st_A.difference_update(st_B) 

print(sum(st_A))
    


#The Captain's Room 
# Enter your code here. Read input from STDIN. Print output to STDOUT

K_sz=int(input())
rooms=list(map(int,input().split()))

s1=set()
s2=set()

for i in rooms:
    if i in s1:
        s2.add(i)
    else:
        s1.add(i)
        
capitain=s1.difference(s2)

print(*capitain)        

#Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
T=int(input())
for i in range(T):
    a=int(input())
    a_set=set(map(int,input().split()))
    b=int(input())
    b_set=set(map(int,input().split()))
    subset=a_set.issubset(b_set)
    print(subset)





#Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
A=set(input().split())
n=int(input())

output=True or False

for i in range(n):
    oth_st=set(input().split())
    if not oth_st.issubset(A):
        output=False
        if len(oth_st) >= len(A):
         output =False
         
print(output)
           

#collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter

n_shoes=int(input())
shoe_sz=list(map(int,input().split()))
customers=int(input())
income=0
shoes=Counter(shoe_sz)
for i in range (customers):
    size,price=map(int,input().split())
    if shoes[size]:
        income+=price
        shoes[size]-=1
print(income)
        
        
    


#DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
d=defaultdict(list)
n,m=list(map(int,input().split()))
for i in range(n):
    d[input()].append(i+1)
for i in range(m):
    print(" ".join(map(str,d[input()]))or -1)
    

     

#Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
n=int(input())
data=namedtuple('data',input())
marks_ls=[]
for i in range (n):
    marks=int(data(*input().split()).MARKS)
    marks_ls.append(marks)
print(sum(marks_ls)/n)   

#Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
n=int(input())
dic={}
for i in range (n):
    lst=list(map(str,input().split()))
    
    if len(lst)>2:
        name=lst[0]+' '+lst[1]
        price=int(lst[-1] )   
    else:
        name=lst[0]
        price=int(lst[1])
        
    if name in dic:
        dic[name] += int(price)
    else:
        dic[name] = int(price)
for name, price in dic.items():
    print(name,price)       
        




#Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict

dic=defaultdict(int)

n=int(input())
for i in range(n):
    dic[input()] +=1
print(len(dic))
print(*dic.values())

#Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque 
d=deque()
n=int(input())

for i in range(n):
    cmnd=input().split()
    if cmnd[0]=="append":
        d.append(cmnd[1])
    elif cmnd[0]=="appendleft":
        d.appendleft(cmnd[1])
    elif cmnd[0]=="pop":
        d.pop()
    elif cmnd[0]=="popleft":
        d.popleft()
    elif cmnd[0]=="popleft":
        d.popleft()
print(*d)        
        

#Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque 
def piling(d):
    while d:
        large=None
        if d[-1] > d[0]:
            large=d.pop()
        else:
            large=d.popleft()
        if len(d)==0:
            return "Yes"
        
        if d[-1]>large or d[0] > large:
            return "No"
for i in range (int(input())):
    no_of_cubes=int(input())
    d=deque(map(int,input().split()))
    print(piling(d))

#Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar 

month,day,year=map(int,input().split())
day=calendar.weekday(year=year,month=month,day=day)
print(calendar.day_name[day].upper())

#Time Delta
#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime as dt

# Complete the time_delta function below.
def time_delta(t1, t2):
    f="%a %d %b % Y %H:%M:%S %z"
    t1=dt.strptime(t1,f)
    t2=dt.strptime(t2,f)
    delta=(t2-t1).total_seconds()
    return abs(int(delta))

for i in range(int(input())):
    print(time_delta(input(),input()))
    

#Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
t=int(input())

for i in range(t):
    try:
        a,b=map(int,input().split()) 
     
        print(a//b)
    except Exception as e:
        print("Error Code:",e)    



#Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
students,subjects=(map(int,input().split()))
marks=[]
for i in range(int(subjects)) :
    votes=list(map(float,input().split())) 
    marks.append(votes)    

for i in zip(*marks):
    print(sum(i)/subjects)


#Map and Lambda Function  ###discuss about this exercise with Enrico Grimaldi, i was not able to do it alone###
cube = lambda x: x**3
def fibonacci(n):
    fibo=[0,1]
    if n==0:
        fibo=[]
    if n==1:
        fibo=[0]
    else:
        for i in range(2,n):  
            fibo.append(fibo[i-1]+fibo[i-2])  
    return fibo            
            
            

       

#Athlete Sort
#!/bin/python3

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    srt_arr=sorted(arr,key=lambda x:x[k]) 
for i in srt_arr:
    print(*i)

#Arrays


def arrays(arr):
    arr=numpy.flip(numpy.array(arr,float))
    return arr


#Shape and Reshape
import numpy
lst=list(map(int,input().split()))

lst_arr=numpy.array(lst)
x=numpy.reshape(lst_arr,(3,3))
print(x)

#Transpose and Flatten
import numpy
r,c= list(map(int,input().split()))
matrix=numpy.array([list(map(int,input().split())) for i in range (r)])
print(matrix.transpose())
print(matrix.flatten())



#Concatenate
import numpy as np

row1,row2,col=map(int,input().split())
row1matrix=[]
row2matrix=[]

for i in range (row1):
    row1matrix.append(np.array(list(map(int,input().split()))))
for i in range (row2):
    row2matrix.append(np.array(list(map(int,input().split()))))
    
print(np.concatenate((row1matrix,row2matrix),axis=0))


#Zeros and Ones
import numpy as np

arr=list(map(int,input().split()))
print(np.zeros(arr,int))
print(np.ones(arr,int))



#Eye and Identity
import numpy as np
np.set_printoptions(legacy='1.13')
print(np.eye(*map(int,input().split())))



#Polynomials
import numpy as np

p=list(map(float,input().split()))
x=float(input())
res=float(np.polyval(p,x))
print(res)

#Linear Algebra
import numpy as np
n=int(input())
A=[list(map(float,input().split())) for i in range(n)]
A=(round(np.linalg.det(A),2))
print(A)






#Array Mathematics
import numpy as np
n,m=list(map(int,input().split()))
a=np.array([(input().split()) for i in range(n)],int )
b=np.array([(input().split()) for i in range(n)],int )

print(np.add(a,b))
print(np.subtract(a,b))
print(np.multiply(a,b))
print(np.floor_divide(a,b))
print(np.mod(a,b))
print(np.power(a,b))


#Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy='1.13')

arr=np.array([input().split()],float)
print(np.floor(*arr))
print(np.ceil(*arr))
print(np.rint(*arr))


#Sum and Prod
import numpy as np
n,m=map(int,input().split())
arr=np.array([list(map(int,input().split()))for i in range(n)])

arr_sum=np.sum(arr,axis=0)
print(np.prod(arr_sum))




#Min and Max
import numpy as np

n,m=map(int,input().split())
matrix=np.array([list(map(int,input().split()))for _ in range(n)],int)
print(np.max(np.min(matrix,axis=1)))




#Mean, Var, and Std
import numpy as np
n,m=map(int,input().split())

matrix=np.array([list(map(int,input().split())) for _ in range(n)],int)

print(np.mean(matrix,axis=1))
print(np.var(matrix,axis=0))
print(round(np.std(matrix),11))

#Dot and Cross
import numpy as np
n=int(input())
A=np.array([list(map(int,input().split()))for _ in range(n)],int)
B=np.array([list(map(int,input().split()))for _ in range(n)],int)
print(np.dot(A,B))

#Inner and Outer
import numpy as np
A=np.array([list(map(int,input().split()))])
B=np.array([list(map(int,input().split()))])
print(np.inner(*A,*B))
print(np.outer(A,B))





#ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
S=list(input())
lower=[i for i in S if i.isalpha() and i.islower()]
upper=[i for i in S if i.isalpha() and i.isupper()]
even=[i for i in S if i.isdigit() and int(i)%2==0]
odd= [i for i in S if i.isdigit()and int(i)%2 != 0]
print("".join(sorted(lower)+sorted(upper)+sorted(odd)+sorted(even)))


#XML 1 - Find the Score


def get_attr_number(node):
    return sum([len(i.keys()) for i in node.iter()])
    



#XML2 - Find the Maximum Depth


maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level==maxdepth):
        maxdepth+=1
    for i in elem:
        depth(i,level+1)



#Standardize Mobile Number Using Decorators   ###helped with an explanation on YT###
def wrapper(f):
    def fun(l):
        cell=lambda x:'+91 '+x[-10:-5]+ ' '+x[-5:]
        f(map(cell,l)) 
    return fun



#Decorators 2 - Name Directory


def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


#Number Line Jumps  ###Discuss about it with Tito Tamburini###

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    if (v1 > v2) and (x2-x1)%(v2-v1)==0:
        return "YES"
    else:
        return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


#Birthday Cake Candles
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    mx=max(candles)
    count=0
    for _ in candles:
        if mx==_:
            count=count+1
    return count
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()




#Re.split()
regex_pattern = r"\W+"	# Do not delete 'r'.


#Viral Advertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    liked=[]
    for i in range (n):
        if i==0:
            liked.append(2)
        else:
            liked.append(liked[i-1]*3//2)
    return sum(liked)
          
       
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()



#Detect Floating Point Number
import re
n=int(input())
for i in range(n):
    if re.search("^[+-]?[0-9]*\.[0-9]+$", input()):
        print("True")
    else:
        print("False")







#Group(), Groups() & Groupdict()
import re
expression=r"([a-zA-Z0-9])\1+"
m=re.search(expression,input())
if m:
    print(m.group(1))
else:
     print(-1)






#Re.findall() & Re.finditer()
import re
#I've seen the solutions on YouTube for this exercise
S=input()
reg_exp=r'(?<=[^aeiou])([aeiou]{2,})(?=[^aeiou])'
matches=re.findall(reg_exp,S,flags=re.IGNORECASE)
if matches:
    for match in matches:
        print(match)
else:
    print(-1)




#Re.start() & Re.end()  ###i've seen an explanation on YT###
import re
S=input()
k=input()
pattern=re.compile(k)
match= pattern.search(S)
if not match:
    print("(-1, -1)")
else:
    while match:
        print("({}, {})".format(match.start(),match.end()-1))
        match=pattern.search(S,match.start()+1)

        
        
#Merge the Tools! ###discuss about it with Tito Tamburiuni, alone i was not able to do it ###
def merge_the_tools(string, k):
    for i in range(0,len(string),k):
        s=string[i:i+k]
        u=set()
        for j in s:
            if j not in u:
                print(j,end='')
                u.add(j)
        print()      
                
    


