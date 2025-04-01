from requests import get

import json
import sqlite3 as sql

def initialize_table():
    cnt = sql.connect('events.db')

    #Create DATE table
    DATE = '''CREATE TABLE IF NOT EXISTS Date (
                Date_ID SERIAL PRIMARY KEY,
                Month INT NOT NULL,
                Day INT NOT NULL,
                Year INT NOT NULL,
                Hour INT NOT NULL,
                Minute INT NOT NULL,
                Second INT NOT NULL,
                UNIQUE (Month, Day, Year, Hour, Minute, Second)
            );'''

    #Create EVENTS table
    EVENTS = '''CREATE TABLE IF NOT EXISTS Events (
                Event_ID SERIAL PRIMARY KEY,
                Event_Name VARCHAR(255) NOT NULL,
                Time TIME NOT NULL
            );'''

    #Create AGENT_COMMANDS table
    AGENT_COMMANDS = '''CREATE TABLE IF NOT EXISTS Agent_Commands (
                        Command_ID SERIAL PRIMARY KEY,
                        Event_ID INT,
                        Command VARCHAR(255) NOT NULL,
                        FOREIGN KEY (Event_ID) REFERENCES Events(Event_ID)
            );'''
    
    #Create SCHEDULE table
    SCHEDULE = '''CREATE TABLE IF NOT EXISTS Schedule (
                    Schedule_ID SERIAL PRIMARY KEY,
                    Date_ID INT,
                    Event_ID INT,
                    Schedule_Order INT,
                    FOREIGN KEY (Date_ID) REFERENCES Dates(Date_ID),
                    FOREIGN KEY (Event_ID) REFERENCES Events(Event_ID)
            );'''

    #Generate each table within the database
    cnt.execute(DATE)
    cnt.execute(EVENTS)
    cnt.execute(AGENT_COMMANDS)
    cnt.execute(SCHEDULE)

    return cnt


def print_table(sens):
    for n in ['Date', 'Events', 'Agent_Commands', 'Schedule']:
        print(n)
        cursor = sens.execute(f'''SELECT * FROM {n};''')
        for i in cursor:
            print(i)


def get_table():
    while True:
        table = int(input('Choose table to add entry:\n1.) Date\n2.) Events\n3.) Agent_Commands\n4.) Schedule\nChoice: '))
        if 1 <= table <= 4:
            return table
        else:
            print('Invalid input')


def add_table(sens):
    t = get_table()
    match t:
        case 1:
            add = input('(Date_ID, Month, Day, Year, Hour, Minute, Second): ')
            name = 'Date'
        case 2:
            add = input('(Event_ID, Event_Name, Time): ')
            name = 'Events'
        case 3:
            add = input('(Command_ID, Event_ID, Command): ')
            name = 'Agent_Commands'
        case 4:
            add = input('(Schedule_ID, Date_ID, Event_ID, Schedule_Order): ')
            name = 'Schedule'
        case _:
            pass

    sens.execute(f'''INSERT INTO {name} VALUES {add}''')


def remove_table(sens):
    t = get_table()
    match t:
        case 1:
            query = input('Choose entry based on Date_ID: ')
            id = 'Date_ID'
            name = 'Date'
        case 2:
            query = input('Choose entry based on Event_ID: ')
            id = 'Event_ID'
            name = 'Events'
        case 3:
            query = input('Choose entry based on Command_ID: ')
            id = 'Command_ID'
            name = 'Agent_Commands'
        case 4:
            query = input('Choose entry based on Schedule_ID: ')
            id = 'Schedule_ID'
            name = 'Schedule'

    sens.execute(f'''DELETE * FROM {name} WHERE {id} = {query}''')


def read_table(sens):
    t = get_table()
    match t:
        case 1:
            query = input('Choose entry based on Date_ID: ')
            id = 'Date_ID'
            name = 'Date'
        case 2:
            query = input('Choose entry based on Event_ID: ')
            id = 'Event_ID'
            name = 'Events'
        case 3:
            query = input('Choose entry based on Command_ID: ')
            id = 'Command_ID'
            name = 'Agent_Commands'
        case 4:
            query = input('Choose entry based on Schedule_ID: ')
            id = 'Schedule_ID'
            name = 'Schedule'

    cursor = sens.execute(f'''SELECT * FROM {name} WHERE {id} = {query}''')
    print(cursor)


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

#REST commands yield 401 errors need to look further
"""
url = 'http://192.168.1.127:8123/api/error_log'
headers = {
    "Authorization": "Bearer TOKEN",
    "content-type": "application/json",
}

response = get(url, headers=headers)
print(response.text)"
"""

homeDB = initialize_table()
access_table(homeDB)
print('Final Table:')
print_table(homeDB)