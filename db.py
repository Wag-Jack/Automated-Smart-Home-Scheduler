import json
import sqlite3 as sql

def initialize_table():
    cnt = sql.connect('events.db')

    #Create DATE table
    DATE = ''' CREATE TABLE DATE (
                Date_ID SERIAL PRIMARY KEY
    
            ); '''
    cnt.execute('''CREATE ''')



    return cnt

def print_table(sens):
    cursor = sens.execute('''SELECT * FROM home_events;''')
    for i in cursor:
        print(str(i[0])+'-'+str(i[1])+'-'+str(i[2]), end=' ') #MM-DD-YYYY
        print(str(i[3])+':'+str(i[4])+':'+str(i[5]), end=' ') #HH:MM:SS
        print("Sensor: "+i[6]+" | State: "+i[7]+' | Agent Prompt: '+i[8],end='\n')

def add_table(sens):
    operation = input().split(' ')

def remove_table(sens):
    operation = input().split(' ')

def read_table(sens):
    pass

def access_table(sens):
    def menu():
        print('Select option:')
        print('1.) Add entry')
        print('2.) Remove entry')
        print('3.) Read entry')
        print('4.) Exit')
        return int(input('Choice: '))
    
    while True:
        match menu():
            case 1:
                add_table(sens)
            case 2:
                remove_table(sens)
            case 3:
                read_table(sens)
            case 4:
                break
            case _:
                print('Invalid input, try again')


homeDB = initialize_table()
access_table(homeDB)
print('Final Table:')
print_table(homeDB)