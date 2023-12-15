import keyboard
import time
import sys
from enum import Enum

arrra= [["🔳","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔳","🔳","🔳","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"],
["🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲","🔲"]]

arra= [[0,0,1,1,1,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0]]

lol= ""
loll= ""

class Movement(Enum):
   DOWN= 1
   RIGHT= 2
   LEFT= 3
   ROTATE= 4
def move(screen: list ,m: Movement) -> list:
   new_arrra= [["🔲"] * 10 for _ in range(10)]
   new_row= 0
   new_column= 0
   for row, value in enumerate(screen):
      for col, v in enumerate(value):
         if v == "🔳":
           match m:
            case Movement.DOWN:
               new_row= row +1
               new_column= col
               print(new_row)
               print("DOWN")  
            case Movement.RIGHT:
               print("DERECHA")    
           new_arrra[new_row][new_column]= "🔳"           
           #print(new_arrra[new_row][new_column])
           #print(len(new_arrra)) 
           #print(str(new_row), str(new_column))      
   draw(new_arrra)
   return new_arrra
   
def draw(lists: list):
   #new_list= lists
   print("new screen")
   for i in lists:
      print("".join(map(str, i)))

def fff():
  #while True:
    screen= move(arrra, Movement.DOWN)    
    screen= move(screen, Movement.DOWN)    
    screen= move(screen, Movement.DOWN)
    #screen= move(screen, Movement.RIGHT)
    #screen= move(screen, Movement.LEFT)
    """if keyboard.is_pressed("q"):
     break

    """
draw(arrra)
fff()

#keyboard.on_press(mecallback,'a')
