import sqlite3

# Paths to the files
SCHEDULE_FILE = "gemini_sensor_schedule.txt"
DB_FILE = "events.db"

# Read the schedule data from file
with open(SCHEDULE_FILE, "r") as f:
    lines = f.readlines()

# Clean and parse each line
records = []
for line in lines:
    line = line.strip()
    if line and not line.startswith("#"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 8:
            try:
                month = int(parts[0])
                day = int(parts[1])
                year = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                second = int(parts[5])
                sensor = parts[6]
                sensor_state = parts[7].upper()

                records.append((month, day, year, hour, minute, second, sensor, sensor_state))
            except ValueError:
                print("Skipping invalid line:", line)

# Insert into SQLite database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.executemany("""
    INSERT INTO home_events (MONTH, DAY, YEAR, HOUR, MINUTE, SECOND, SENSOR, SENSOR_STATE)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", records)

conn.commit()
conn.close()

print(f"Inserted {len(records)} rows into home_events table.")
